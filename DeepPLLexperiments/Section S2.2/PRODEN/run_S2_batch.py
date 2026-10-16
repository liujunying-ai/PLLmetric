import os
import re
import time
import subprocess
import numpy as np
from scipy.io import savemat

# ==================== 配置 ====================
DATASETS = {
    'lost':     {'lr': '1e-2', 'wd': '1e-2', 'bs': '100', 'ep': '200'},
    'MSRCv2':   {'lr': '1e-1', 'wd': '1e-3', 'bs': '100', 'ep': '200'},
    'BirdSong': {'lr': '1e-2', 'wd': '1e-4', 'bs': '100', 'ep': '200'},
}
MODELS  = ['linear', 'mlp']     # 想只跑 linear 就改成 ['linear']
N_RUNS  = 10                    # 每个配置重复跑的次数
FOLD    = 0                     # 用十折划分的哪一折（0 = 第 1 折）
SILENT  = True
# ===============================================

results_dir = './results_S2/'
os.makedirs(results_dir, exist_ok=True)
DEVNULL = subprocess.DEVNULL if SILENT else None


def run_once(dataset, cfg, model, oracle, seed):
    """跑一次 main.py，返回 teACC / disACCt / disACCd。oracle 模式下 disACCt/disACCd 恒为 100。"""
    before = set(os.listdir(results_dir))
    cmd = ['python', 'main.py',
           '-ds', dataset, '-model', model,
           '-lr', cfg['lr'], '-wd', cfg['wd'],
           '-bs', cfg['bs'], '-ep', cfg['ep'],
           '-nw', '0',
           '-seed', str(seed),
           '-dir', results_dir]
    if oracle:
        cmd.append('-oracle')
    subprocess.run(cmd, check=True, stdout=DEVNULL, stderr=DEVNULL)

    after = set(os.listdir(results_dir))
    new_files = list(after - before)
    if not new_files:
        raise RuntimeError(f'{dataset}/{model}: 没有新结果文件生成')
    new_file = os.path.join(results_dir, new_files[0])
    with open(new_file, 'r') as f:
        text = f.read()

    def grab(key):
        m = re.search(rf'{key}\s*=\s*([\d.]+)', text)
        return float(m.group(1)) if m else float('nan')

    return {'teACC': grab('teACC'),
            'disACCt': grab('disACCt'),
            'disACCd': grab('disACCd')}


print('=' * 72)
print(f'PRODEN S2 批量实验：三个数据集一起跑')
print(f'数据集: {list(DATASETS.keys())}')
print(f'主干: {MODELS}, 每个配置跑 {N_RUNS} 次, fold={FOLD}')
print('=' * 72)

for dataset, cfg in DATASETS.items():
    for model in MODELS:
        print(f'\n########## {dataset} / {model} ##########')
        te_list, ora_list, dt_list, dd_list = [], [], [], []

        for i in range(N_RUNS):
            seed = 1000 + i * 37   # 每次不同但固定的种子
            print(f'  Run {i+1}/{N_RUNS} (seed={seed})')

            rp = run_once(dataset, cfg, model, oracle=False, seed=seed)
            ro = run_once(dataset, cfg, model, oracle=True,  seed=seed)

            te_list.append(rp['teACC'])
            dt_list.append(rp['disACCt'])
            dd_list.append(rp['disACCd'])
            ora_list.append(ro['teACC'])

            print(f"    [partial] teACC={rp['teACC']:.4f}  "
                  f"disACCt={rp['disACCt']:.4f}  disACCd={rp['disACCd']:.4f}")
            print(f"    [oracle ] oraACC={ro['teACC']:.4f}")

        mat_dict = {
            'teACC':   np.array(te_list),
            'oraACC':  np.array(ora_list),
            'disACCt': np.array(dt_list),
            'disACCd': np.array(dd_list),
            'dataset': dataset, 'model': model,
            'n_runs': N_RUNS, 'fold': FOLD,
        }
        mat_dict['disGap'] = mat_dict['oraACC'] - mat_dict['teACC']
        mat_dict['relAcc'] = mat_dict['teACC'] / mat_dict['oraACC']

        out = os.path.join(results_dir, f'PRODEN_S2_{dataset}_{model}.mat')
        savemat(out, mat_dict, do_compression=True)
        print(f'  已保存: {out}')
        print(f'  teACC mean={mat_dict["teACC"].mean():.4f}, '
              f'oraACC mean={mat_dict["oraACC"].mean():.4f}, '
              f'relAcc mean={mat_dict["relAcc"].mean():.4f}')

print('\n全部完成。')