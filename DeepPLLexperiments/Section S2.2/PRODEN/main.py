import os
import os.path
import torch 
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms 
import argparse
import numpy as np
import time

from utils.utils_loss import partial_loss
from utils.models import linear, mlp
from cifar_models import convnet, resnet
from datasets.mnist import mnist
from datasets.fashion import fashion
from datasets.kmnist import kmnist
from datasets.cifar10 import cifar10
from datasets.partial_mat import PartialMatDataset

# torch.manual_seed(0); torch.cuda.manual_seed_all(0)

parser = argparse.ArgumentParser(
	prog='PRODEN demo file.',
	usage='Demo with partial labels.',
	description='A simple demo file with MNIST dataset.',
	epilog='end',
	add_help=True)

parser = argparse.ArgumentParser()
parser.add_argument('-lr', help='optimizer\'s learning rate', type=float, default=1e-3)
parser.add_argument('-wd', help='weight decay', type=float, default=1e-5)
parser.add_argument('-bs', help='batch size', type=int, default=256)
parser.add_argument('-ep', help='number of epochs', type=int, default=500)
parser.add_argument('-ds', help='specify a dataset', type=str, default='mnist', choices=['mnist', 'fashion', 'kmnist', 'cifar10', 'BirdSong', 'FG-NET', 'italian', 'lost', 'malagasy','Mirflickr', 'MSRCv2', 'Soccer', 'Yahoo'], required=False)
parser.add_argument('-model', help='model name', type=str, default='linear', choices=['linear', 'mlp', 'convnet', 'resnet'], required=False)
parser.add_argument('-decaystep', help='learning rate\'s decay step', type=int, default=500)
parser.add_argument('-decayrate', help='learning rate\'s decay rate', type=float, default=1)

parser.add_argument('-partial_type', help='flipping strategy', type=str, default='binomial', choices=['binomial', 'pair'])
parser.add_argument('-partial_rate', help='flipping probability', type=float, default=0.1)

parser.add_argument('-nw', help='multi-process data loading', type=int, default=4, required=False)
parser.add_argument('-dir', help='result save path', type=str, default='results/', required=False)
parser.add_argument('-oracle', help='use true labels instead of partial labels', action='store_true')
parser.add_argument('-seed', type=int, default=42)
parser.add_argument('-fold', type=int, default=0)

args = parser.parse_args()

# 设置随机种子（每次不同，用于多次运行）
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# load dataset
if args.ds == 'mnist':
    num_features = 28 * 28
    num_classes = 10
    num_training = 60000
    train_dataset = mnist(root='./mnist/',
                                download=True,  
                                train_or_not=True, 
                                transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]), 
                                partial_type=args.partial_type,
                                partial_rate=args.partial_rate
    )
    test_dataset = mnist(root='./mnist/',
                               download=True,  
                               train_or_not=False, 
                               transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]),
                               partial_type=args.partial_type,
                               partial_rate=args.partial_rate
    )

if args.ds == 'fashion':
    num_features = 28 * 28
    num_classes = 10
    num_training = 60000
    train_dataset = fashion(root='./fashion/',
                                download=True,  
                                train_or_not=True, 
                                transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]), 
                                partial_type=args.partial_type,
                                partial_rate=args.partial_rate
    ) 
    test_dataset = fashion(root='./fashionmnist/',
                               download=True,  
                               train_or_not=False, 
                               transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]),
                               partial_type=args.partial_type,
                               partial_rate=args.partial_rate
    )

if args.ds == 'kmnist':
    num_features = 28 * 28
    num_classes = 10
    num_training = 60000
    train_dataset = kmnist(root='./kmnist/',
                                download=True,  
                                train_or_not=True, 
                                transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))]), 
                                partial_type=args.partial_type,
                                partial_rate=args.partial_rate
    ) 
    test_dataset = kmnist(root='./kmnist/',
                               download=True,  
                               train_or_not=False, 
                               transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))]), 
                               partial_type=args.partial_type,
                               partial_rate=args.partial_rate
    )

if args.ds == 'cifar10':
    input_channels = 3
    num_classes = 10
    dropout_rate = 0.25
    num_training = 50000
    train_dataset = cifar10(root='./cifar10/',
                                download=True,  
                                train_or_not=True, 
                                transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),]),
                                partial_type=args.partial_type,
                                partial_rate=args.partial_rate
    ) 
    test_dataset = cifar10(root='./cifar10/',
                                download=True,  
                                train_or_not=False, 
                                transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),]),
                                partial_type=args.partial_type,
                                partial_rate=args.partial_rate
    )

if args.ds in ['BirdSong', 'FG-NET', 'italian', 'lost', 'malagasy',
               'Mirflickr', 'MSRCv2', 'Soccer', 'Yahoo']:
    data_root = './data/'
    train_dataset = PartialMatDataset(root=data_root, dataset_name=args.ds,
                                       train_or_not=True,  fold=args.fold)
    test_dataset  = PartialMatDataset(root=data_root, dataset_name=args.ds,
                                       train_or_not=False, fold=args.fold)
    num_features = train_dataset.num_features
    num_classes  = train_dataset.num_classes
    print(f'[{args.ds}] num_features={num_features}, num_classes={num_classes}, '
          f'train={len(train_dataset)}, test={len(test_dataset)}')
		  
    # ===== 新增：oracle 模式 =====
    if args.oracle:
        train_dataset.train_final_labels = train_dataset.true_onehot.clone()
        train_dataset.partial = train_dataset.true_onehot.clone()
        print('[oracle] 已用真实标签替换候选标签')

# learning rate 
lr_plan = [args.lr] * args.ep 
for i in range(0, args.ep):
    lr_plan[i] = args.lr * args.decayrate ** (i // args.decaystep)
def adjust_learning_rate(optimizer, epoch):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_plan[epoch]


# result dir  
save_dir = './' + args.dir
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 生成时间戳，例如 2026_10_09_04_02_30
time_str = time.strftime('%Y_%m_%d_%H_%M_%S')

# 把命令行的关键参数也加进文件名
# 用 :g 让浮点数显示得简洁一些（0.01 而不是 0.010000000000000002）
filename = (
    f'{args.ds}'
    f'_model{args.model}'
    f'_lr{args.lr:g}'
    f'_wd{args.wd:g}'
    f'_bs{args.bs}'
    f'_ep{args.ep}'
    f'_{time_str}.txt'
)
save_file = os.path.join(save_dir, filename)

# calculate accuracy
def evaluate(loader, model):
    model.eval()     
    correct = 0
    total = 0
    for images, _, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device)
        output1 = model(images)
        output = F.softmax(output1, dim=1)
        _, pred = torch.max(output.data, 1) 
        total += images.size(0)
        correct += (pred == labels).sum().item()
    acc = 100 * float(correct) / float(total)
    return acc


def evaluate_dis_t(loader, model):
    """训练集消歧精度：用模型预测，但限制在候选标签内选最大。"""
    model.eval()
    correct = 0
    total = 0
    for images, labels, trues, _ in loader:
        images = images.to(device)
        labels = labels.to(device)   # 候选标签权重，>0 表示在候选集内
        trues  = trues.to(device)
        output = model(images)
        # 把非候选位置设成一个很小的数，只在候选内选最大
        mask = (labels > 0).float()
        output = output - (1 - mask) * 1e9
        _, pred = torch.max(output.data, 1)
        total += images.size(0)
        correct += (pred == trues).sum().item()
    return 100 * float(correct) / float(total)


def evaluate_dis_d(loader):
    """训练集消歧精度：直接用 train_final_labels 的 argmax。"""
    dataset = loader.dataset
    final_labels = dataset.train_final_labels  # (n_train, n_classes)
    trues = dataset.true_labels                # (n_train,)
    pred = final_labels.argmax(dim=1)
    correct = (pred == trues).sum().item()
    return 100 * float(correct) / len(trues)


def main():
    # print ('loading dataset...')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.bs, 
                                               num_workers=args.nw,
                                               drop_last=True, 
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.bs, 
                                              num_workers=args.nw,
                                              drop_last=False,
                                              shuffle=False)

    # print ('building model...')
    if args.model == 'linear':
        net = linear(n_inputs=num_features, n_outputs=num_classes)
    elif args.model == 'mlp':
        net = mlp(n_inputs=num_features, n_outputs=num_classes)
    elif args.model == 'convnet':
        net = convnet(input_channels=input_channels, n_outputs=num_classes, dropout_rate=dropout_rate)
    elif args.model == 'resnet':
        net = resnet(depth=32, n_outputs=num_classes)
    net.to(device)
    print (net.parameters)

    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9)

    for epoch in range(0, args.ep):
        # print ('training...')
        net.train()
        adjust_learning_rate(optimizer, epoch)

        for i, (images, labels, trues, indexes) in enumerate(train_loader): 
            images = Variable(images).to(device)
            labels = Variable(labels).to(device)
            trues = trues.to(device)
            output = net(images)
            
            loss, new_label = partial_loss(output, labels, trues)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update weights
            for j, k in enumerate(indexes):
                train_loader.dataset.train_final_labels[k,:] = new_label[j,:].detach()
 
        # print ('evaluating model...')       
        train_acc = evaluate(train_loader, net)
        test_acc = evaluate(test_loader, net)

        with open(save_file, 'a') as file:
            file.write(str(int(epoch)) + ': Training Acc.: ' + str(round(train_acc, 4)) + ' , Test Acc.: ' + str(round(test_acc, 4)) + '\n')
			
    # ===== 训练结束后计算最终指标 =====
    final_te_acc   = evaluate(test_loader, net)
    final_dis_t    = evaluate_dis_t(train_loader, net)
    final_dis_d    = evaluate_dis_d(train_loader)

    print('\n========== 最终结果 ==========')
    print(f'teACC   (测试集精度)          : {final_te_acc:.4f}')
    print(f'disACCt (训练集消歧, 模型预测) : {final_dis_t:.4f}')
    print(f'disACCd (训练集消歧, 软标签)   : {final_dis_d:.4f}')

    # 写到结果文件末尾
    with open(save_file, 'a') as f:
        f.write('\n===== FINAL =====\n')
        f.write(f'teACC={final_te_acc:.4f}\n')
        f.write(f'disACCt={final_dis_t:.4f}\n')
        f.write(f'disACCd={final_dis_d:.4f}\n')
    

if __name__=='__main__':
    main()
