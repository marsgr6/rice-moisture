"""Proxy-cost versus measured-cost DEA on identical pipelines, data, and performance estimates.
Usage: python3 -I dea_compare.py scores.csv features15.npz OUTDIR
DMU = classifier x descriptor of the re-evaluation (15% scale, 1630 photographs).
Outputs (both models): mean time-block macro F1 over the 8 conditions and stability 1 - CV across conditions.
Inputs: proxy = feature length of the re-implementation (the scale is the same for all DMUs);
measured = median extraction time per image (ms) and median serialized model size (kB).
Writes OUTDIR/dea_compare.csv and prints the comparison."""
import os, sys, numpy as np, pandas as pd
from scipy.stats import spearmanr, kendalltau
from scipy.optimize import linprog
D = ['CMMS', 'G_H', 'RGB', 'HSV', 'LBP', 'GLCM', 'HRLK']


def dea(X, Y, vrs=False):
    """Input-oriented radial DEA (CCR, or BCC if vrs); returns theta for each DMU."""
    n = len(X); th = []
    for o in range(n):
        c = np.r_[1, np.zeros(n)]
        A = [np.r_[-X[o, i], X[:, i]] for i in range(X.shape[1])] + [np.r_[0, -Y[:, r]] for r in range(Y.shape[1])]
        b = [0] * X.shape[1] + list(-Y[o])
        kw = dict(A_eq=[np.r_[0, np.ones(n)]], b_eq=[1]) if vrs else {}
        th.append(linprog(c, A_ub=A, b_ub=b, bounds=[(None, None)] + [(0, None)] * n, method='highs', **kw).x[0])
    return np.array(th)

if __name__ == '__main__':
    s = pd.read_csv(sys.argv[1]); F = np.load(sys.argv[2]); out = sys.argv[3]
    s = s[(s.variant == 'orig') & s.desc.isin(D)]
    pd.set_option('display.width', 200)
    rows = []
    for proto in ['block', 'random']:
        q = s[s.proto == proto]
        f = q.groupby(['clf', 'desc', 'cond']).f1.mean().unstack('cond')
        t = pd.DataFrame({'F1': f.mean(1), 'stab': 1 - f.std(1, ddof=1) / f.mean(1)})
        desc = t.index.get_level_values('desc')
        t['length'] = [F[f'orig_{d}'].shape[1] for d in desc]
        t['ext_ms'] = [np.median(F[f'time_{d}']) * 1000 for d in desc]
        t['size_kB'] = q.groupby(['clf', 'desc'])['size'].median() / 1024
        Y = t[['F1', 'stab']].values
        t['CRS_proxy'] = dea(t[['length']].values.astype(float), Y)
        t['CRS_meas'] = dea(t[['ext_ms', 'size_kB']].values, Y)
        t['CRS_ext'] = dea(t[['ext_ms']].values, Y)
        for k in ['proxy', 'meas', 'ext']:
            t[f'rank_{k}'] = (t[f'CRS_{k}'] * t.F1).rank(ascending=False, method='min').astype(int)
        t['proto'] = proto; rows.append(t.reset_index())
        print(f'=== protocol {proto}: 21 DMUs, identical outputs ===')
        print(t.sort_values('rank_proxy').drop(columns='proto').round(3))
        print('descriptor means of CRS:')
        print(t.groupby(level='desc')[['length', 'ext_ms', 'CRS_proxy', 'CRS_meas', 'CRS_ext']].mean()
              .sort_values('CRS_proxy', ascending=False).round(3))
        for a, b in [('proxy', 'meas'), ('proxy', 'ext'), ('meas', 'ext')]:
            print(f'Spearman CRS {a} vs {b}: {spearmanr(t[f"CRS_{a}"], t[f"CRS_{b}"]).correlation:.2f}, '
                  f'Kendall rank(CRS x F1): {kendalltau(t[f"rank_{a}"], t[f"rank_{b}"]).correlation:.2f}')
        for k in ['proxy', 'meas']:
            fr = t[t[f'CRS_{k}'] > 0.9999].index.tolist()
            print(f'frontier {k}: {fr}')
            print(f'top5 {k}:', t.sort_values(f'rank_{k}').index[:5].tolist())
        print()
    pd.concat(rows).to_csv(os.path.join(out, 'dea_compare.csv'), index=False)
