"""Measured-cost DEA of the re-run (Section 3.7 / 4.4).
Usage: python3 -I dea_measured.py scores.csv features15.npz
DMU = classifier x descriptor (15% scale). Inputs: median extraction time per image (ms),
median serialized model size (kB). Outputs: mean macro F1 over the 8 conditions and
stability 1 - CV of F1 across conditions. Input-oriented CCR (CRS) and BCC (VRS)."""
import sys, numpy as np, pandas as pd
from scipy.optimize import linprog
pd.set_option('display.width', 200)
D = ['CMMS', 'G_H', 'RGB', 'HSV', 'LBP', 'GLCM', 'HRLK']

def dea(X, Y, vrs=False):
    n = len(X); th = []
    for o in range(n):
        c = np.r_[1, np.zeros(n)]
        A = [np.r_[-X[o, i], X[:, i]] for i in range(X.shape[1])] + [np.r_[0, -Y[:, r]] for r in range(Y.shape[1])]
        b = [0] * X.shape[1] + list(-Y[o])
        kw = dict(A_eq=[np.r_[0, np.ones(n)]], b_eq=[1]) if vrs else {}
        th.append(linprog(c, A_ub=A, b_ub=b, bounds=[(None, None)] + [(0, None)] * n, method='highs', **kw).x[0])
    return np.array(th)

s = pd.read_csv(sys.argv[1]); F = np.load(sys.argv[2])
s = s[(s.variant == 'orig') & (s.desc.isin(D))]
ext = {d: np.median(F[f'time_{d}']) * 1000 for d in D}
for proto in ['block', 'random']:
    for name, conds in [('all conditions', None), ('chamber only', ['C1', 'C2', 'C3', 'C4', 'C8'])]:
        q = s[s.proto == proto] if conds is None else s[(s.proto == proto) & s.cond.isin(conds)]
        f = q.groupby(['clf', 'desc', 'cond']).f1.mean().unstack('cond')
        t = pd.DataFrame({'F1': f.mean(1), 'stab': 1 - f.std(1, ddof=1) / f.mean(1)})
        t['ext_ms'] = [ext[d] for d in t.index.get_level_values('desc')]
        t['size_kB'] = q.groupby(['clf', 'desc'])['size'].median() / 1024
        X, Y = t[['ext_ms', 'size_kB']].values, t[['F1', 'stab']].values
        t['CRS'] = dea(X, Y); t['VRS'] = dea(X, Y, True); t['score'] = t.CRS * t.F1
        if proto == 'block' and conds is None:
            t.reset_index().to_csv('dea_measured_block.csv', index=False)
        print(f'=== protocol {proto}, {name}: inputs extraction ms/image + model size kB ===')
        print(t.sort_values('score', ascending=False).round(3))
        print(t.groupby(level='desc')[['F1', 'stab', 'CRS']].mean().sort_values('CRS', ascending=False).round(3))
