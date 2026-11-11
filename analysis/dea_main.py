"""DEA of the 126 global-descriptor pipelines (3 classifiers x 7 descriptors x 6 scales).

Inputs : feature vector length of the descriptor (as produced by the notebooks) and image scale (%).
Outputs: mean macro F1-score over the eight acquisition conditions, and
         stability = 1 - coefficient of variation of the F1-score across conditions.
Models : input-oriented CCR (CRS) and BCC (VRS), second-stage additive slacks (Pareto efficiency),
         and sensitivity analyses (leave one condition out, bootstrap over conditions,
         chamber conditions only, alternative ranking rules).

Usage: python3 dea_main.py RESULTS/GLOBAL/EV/mergedEV.csv OUTDIR
Writes OUTDIR/dea_results.csv and prints the tables of the paper.
"""
import os, sys
import numpy as np, pandas as pd
from scipy.optimize import linprog

# Feature vector length produced by the notebooks:
# CMMS: mean, standard deviation and third central moment of H, S, V (9)
# HRLK: 13 Haralick features (mahotas) in each of 4 directions, concatenated (52)
# GLCM: 6 properties (contrast, dissimilarity, homogeneity, energy, correlation, ASM) x 4 directions (24)
# LBP : histogram of uniform LBP codes, P = 16, R = 2 (18)
# G_H : grayscale histogram, 44 bins (44)
# RGB, HSV: 3-D color histogram, 16 x 16 x 16 bins (4096)
LENGTH = {'CMMS': 9, 'LBP': 18, 'GLCM': 24, 'G_H': 44, 'HRLK': 52, 'RGB': 4096, 'HSV': 4096}
CHAMBER = ['C01', 'C02', 'C03', 'C04', 'C08']


def load(path):
    d = pd.read_csv(path, encoding='utf-8-sig').dropna(subset=['model'])
    m = d[d.group == 'macro avg']
    return m.pivot_table(index=['model', 'algorithm', 'scale'], columns='data', values='f1-score')


def outputs(P):
    mu = P.mean(1); sd = P.std(1, ddof=1)
    return mu, 1 - sd / mu


def inputs(P):
    return np.c_[[LENGTH[a] for a in P.index.get_level_values(1)], P.index.get_level_values(2)].astype(float)


def dea(X, Y, vrs=False):
    """Input-oriented radial efficiency of every DMU."""
    n = len(X); th = []
    for o in range(n):
        c = np.r_[1, np.zeros(n)]
        A = [np.r_[-X[o, i], X[:, i]] for i in range(X.shape[1])] + [np.r_[0, -Y[:, r]] for r in range(Y.shape[1])]
        b = [0] * X.shape[1] + list(-Y[o])
        kw = dict(A_eq=[np.r_[0, np.ones(n)]], b_eq=[1]) if vrs else {}
        th.append(linprog(c, A_ub=A, b_ub=b, bounds=[(None, None)] + [(0, None)] * n, method='highs', **kw).x[0])
    return np.array(th)


def slacks(X, Y, th, vrs=False):
    """Second-stage additive model: maximum sum of slacks given theta."""
    n, m_, s_ = len(X), X.shape[1], Y.shape[1]; out = []
    for o in range(n):
        c = np.r_[np.zeros(n), -np.ones(m_ + s_)]
        Aeq, beq = [], []
        for i in range(m_):
            e = np.zeros(m_ + s_); e[i] = 1; Aeq.append(np.r_[X[:, i], e]); beq.append(th[o] * X[o, i])
        for r in range(s_):
            e = np.zeros(m_ + s_); e[m_ + r] = -1; Aeq.append(np.r_[Y[:, r], e]); beq.append(Y[o, r])
        if vrs:
            Aeq.append(np.r_[np.ones(n), np.zeros(m_ + s_)]); beq.append(1)
        out.append(-linprog(c, A_eq=Aeq, b_eq=beq, bounds=[(0, None)] * (n + m_ + s_), method='highs').fun)
    return np.array(out)


def run(P):
    mu, st = outputs(P); X = inputs(P); Y = np.c_[mu.values, st.values]
    r = pd.DataFrame(index=P.index)
    r['length'] = X[:, 0]; r['f1'] = mu.values; r['stability'] = st.values
    r['crs'] = dea(X, Y); r['vrs'] = dea(X, Y, vrs=True)
    return r


def top(r, key, k=5):
    return r.assign(score=key(r)).sort_values('score', ascending=False).head(k)


if __name__ == '__main__':
    src, outdir = sys.argv[1:3]; os.makedirs(outdir, exist_ok=True)
    P = load(src); r = run(P)
    X = inputs(P); Y = r[['f1', 'stability']].values
    r['slack_vrs'] = slacks(X, Y, r.vrs.values, vrs=True)
    r['pareto'] = (r.vrs > 1 - 1e-6) & (r.slack_vrs < 1e-6)
    r['crs_x_f1'] = r.crs * r.f1
    r.sort_values('crs_x_f1', ascending=False).to_csv(os.path.join(outdir, 'dea_results.csv'))
    pd.set_option('display.width', 200)
    print('=== Top five by theta_CRS x F1'); print(top(r, lambda d: d.crs * d.f1).round(4))
    print('\n=== By classifier'); print(r.groupby(level=0).agg(mean_crs=('crs', 'mean'), best_crs=('crs', 'max'), mean_f1=('f1', 'mean'), frontier=('crs', lambda x: (x > 1 - 1e-6).sum())).round(4))
    print('\n=== By descriptor'); print(r.groupby(level=1)[['length', 'crs', 'vrs', 'f1', 'stability']].mean().sort_values('crs', ascending=False).round(4))
    print('\n=== Range of mean CRS across classifiers within each descriptor')
    g = r.groupby(level=[1, 0]).crs.mean().unstack(); print((g.max(1) - g.min(1)).round(3).to_dict())
    print('\n=== CRS frontier'); print(r[r.crs > 1 - 1e-6].sort_values('f1', ascending=False).round(4))
    print('\nVRS-efficient:', int((r.vrs > 1 - 1e-6).sum()), ' Pareto (VRS):', int(r.pareto.sum()))
    # sensitivity
    print('\n=== Sensitivity: descriptors of the top five (theta_CRS x F1)')
    key = lambda d: d.crs * d.f1
    print('Original:', top(r, key).index.get_level_values(1).tolist())
    for c in P.columns:
        print(' without', c, ':', top(run(P.drop(columns=c)), key).index.get_level_values(1).tolist())
    rng = np.random.default_rng(0); counts = []; first = []
    for b in range(200):
        cols = rng.choice(P.columns, len(P.columns), replace=True)
        Pb = P[cols]; Pb.columns = range(len(cols))
        t = top(run(Pb), key).index.get_level_values(1)
        counts.append(pd.Series(t).value_counts()); first.append(t[0])
    print('Bootstrap (200): mean count in top five', pd.concat(counts, axis=1).fillna(0).mean(1).round(2).to_dict())
    print('Bootstrap (200): share ranked first', (pd.Series(first).value_counts() / 200).round(2).to_dict())
    rc = run(P[CHAMBER])
    print('Chamber only:', top(rc, key).index.tolist())
    print('Chamber only, mean CRS by descriptor', rc.groupby(level=1).crs.mean().round(2).to_dict())
    print('Rank by CRS x stability:', top(r, lambda d: d.crs * d.stability).index.tolist())
    print('Rank by VRS x F1:', top(r, lambda d: d.vrs * d.f1).index.tolist())
    print('Rank by CRS:', top(r, lambda d: d.crs + 1e-6 * d.f1).index.tolist())
