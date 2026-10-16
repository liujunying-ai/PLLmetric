import os
import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F

from models.model_linear import Linearnet
from models.model_mlp import Mlp
from utils.data_real import RealPartialDataset
from utils.utils_algo import confidence_update, confidence_update_lw
from utils.utils_loss import lws_loss, lwc_loss, cc_loss, rc_loss


def accuracy_check(loader, model, device):
    """普通精度：在全部预测里取最大。"""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, _, y, _ in loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            _, pred = torch.max(out.data, 1)
            correct += (pred == y).sum().item()
            total += X.size(0)
    return 100.0 * correct / total


def disamb_acc(loader, model, device):
    """消歧精度：把非候选位置屏蔽掉，只在候选内取最大。"""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, partial, y, _ in loader:
            X, partial, y = X.to(device), partial.to(device), y.to(device)
            out = model(X)
            mask = (partial > 0).float()
            out = out - (1 - mask) * 1e9
            _, pred = torch.max(out.data, 1)
            correct += (pred == y).sum().item()
            total += X.size(0)
    return 100.0 * correct / total


def main():
    p = argparse.ArgumentParser()
    p.add_argument('-ds', type=str, default='lost')
    p.add_argument('-mo', type=str, default='linear', choices=['linear', 'mlp'])
    #p.add_argument('-lo', type=str, default='lws', choices=['lws', 'cc', 'rc'])
    p.add_argument('-lo', type=str, default='lwc', choices=['lws', 'lwc', 'cc', 'rc'])
    p.add_argument('-lw', type=float, default=1.0)      # beta (非候选损失权重)
    p.add_argument('-lw0', type=float, default=1.0)     # alpha (候选损失权重)
    p.add_argument('-lr', type=float, default=1e-2)
    p.add_argument('-wd', type=float, default=1e-4)
    p.add_argument('-bs', type=int, default=100)
    p.add_argument('-ep', type=int, default=200)
    p.add_argument('-seed', type=int, default=42)
    p.add_argument('-fold', type=int, default=0)
    p.add_argument('-gpu', type=str, default='0')
    p.add_argument('-oracle', action='store_true')
    p.add_argument('-ldr', type=float, default=0.5)   # 学习率衰减率
    p.add_argument('-lds', type=int, default=50)      # 每多少 epoch 衰减一次
    p.add_argument('-dir', type=str, default='./results_lw/')
    args = p.parse_args()

    device = torch.device('cuda:' + args.gpu if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_root = './data/'
    train_dataset = RealPartialDataset(data_root, args.ds, train_or_not=True,
                                       fold=args.fold)
    test_dataset  = RealPartialDataset(data_root, args.ds, train_or_not=False,
                                       fold=args.fold)

    K = train_dataset.num_classes
    dim = train_dataset.num_features
    n = len(train_dataset)

    # ===== oracle 模式：用真实标签替换候选标签 =====
    if args.oracle:
        onehot = torch.zeros(n, K)
        onehot[torch.arange(n), train_dataset.true_labels] = 1.0
        train_dataset.partial = onehot
        print('[oracle] 已用真实标签替换候选标签')

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.bs, shuffle=True,
        drop_last=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.bs, shuffle=False,
        drop_last=False, num_workers=0)

    # ===== 模型 =====
    if args.mo == 'linear':
        model = Linearnet(n_inputs=dim, n_outputs=K)
    else:
        model = Mlp(n_inputs=dim, n_outputs=K)
    model = model.to(device)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9)

    lr_plan = [args.lr * (args.ldr ** (i // args.lds)) for i in range(args.ep)]
    def adjust_lr(optimizer, epoch):
        for g in optimizer.param_groups:
            g['lr'] = lr_plan[epoch]

    # ===== 置信度初始化 =====
    #if not args.oracle and args.lo in ('lws', 'rc'):
    #    confidence = (train_dataset.partial.clone()).to(device)
    #else:
    #    confidence = None
    if not args.oracle and args.lo in ('lws', 'lwc'):
        # LW 要求：候选内归一化到 1，非候选内归一化到 1（每行和=2）
        partial = train_dataset.partial
        cand = (partial > 0).float()
        K_dim = partial.shape[1]
        cand_count = cand.sum(dim=1, keepdim=True)              # |y_i|
        non_count = K_dim - cand_count                          # K - |y_i|
        conf = cand / (cand_count + 1e-8) + (1 - cand) / (non_count + 1e-8)
        confidence = conf.to(device)
    elif not args.oracle and args.lo == 'rc':
        confidence = train_dataset.partial.clone().to(device)
    else:
        confidence = None
		
    # ===== 结果文件 =====
    os.makedirs(args.dir, exist_ok=True)
    time_str = time.strftime('%Y_%m_%d_%H_%M_%S')
    tag = 'oracle' if args.oracle else f'{args.lo}-lw{args.lw:g}'
    fname = (f'{args.ds}_{args.mo}_{tag}'
             f'_lr{args.lr:g}_wd{args.wd:g}_ep{args.ep}'
             f'_{time_str}.txt')
    save_file = os.path.join(args.dir, fname)

    # ===== 训练循环 =====
    for epoch in range(args.ep):
        adjust_lr(optimizer, epoch)
        model.train()
        for X, Y, _, index in train_loader:
            X, Y, index = X.to(device), Y.to(device), index.to(device)
            optimizer.zero_grad()
            out = model(X)

            if args.oracle:
                # oracle 模式：直接用交叉熵
                loss = cc_loss(out, Y)
            elif args.lo == 'lws':
                loss, _, _ = lws_loss(out, Y, confidence, index,
                                      args.lw, args.lw0, None)
            elif args.lo == 'lwc':
                loss, _, _ = lwc_loss(out, Y, confidence, index,
                                      args.lw, args.lw0, None)
            elif args.lo == 'cc':
                loss = cc_loss(out, Y)
            elif args.lo == 'rc':
                logsm = F.log_softmax(out, dim=1)
                loss = -((logsm * confidence[index, :]).sum(dim=1)).mean()

            loss.backward()
            optimizer.step()

            if not args.oracle:
                if args.lo in ('lws', 'lwc'):
                    confidence = confidence_update_lw(model, confidence, X, Y, index)
                elif args.lo == 'rc':
                    confidence = confidence_update(model, confidence, X, Y, index)

        if (epoch + 1) % 20 == 0 or epoch == args.ep - 1:
            tr = accuracy_check(train_loader, model, device)
            te = accuracy_check(test_loader, model, device)
            print(f'Epoch {epoch+1:>4d}: TrAcc={tr:.4f}  TeAcc={te:.4f}')

    # ===== 最终指标 =====
    te_acc = accuracy_check(test_loader, model, device)
    dis_t = disamb_acc(train_loader, model, device)

    if (not args.oracle) and args.lo in ('lws', 'rc'):
        pred = confidence.argmax(dim=1).cpu()
        true = train_dataset.true_labels
        dis_d = 100.0 * (pred == true).sum().item() / n
    else:
        dis_d = dis_t   # oracle 模式下两者等价

    print('\n========== FINAL ==========')
    print(f'teACC   = {te_acc:.4f}')
    print(f'disACCt = {dis_t:.4f}')
    print(f'disACCd = {dis_d:.4f}')

    with open(save_file, 'w') as f:
        f.write(f'teACC={te_acc:.4f}\n')
        f.write(f'disACCt={dis_t:.4f}\n')
        f.write(f'disACCd={dis_d:.4f}\n')


if __name__ == '__main__':
    main()