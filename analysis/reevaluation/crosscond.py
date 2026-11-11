"""Cross-condition transfer: train on all images of one acquisition condition, test on every other condition.
Usage: python3 -I crosscond.py MANIFEST FEATURES.npz SCORES.csv OUTDIR
Writes OUTDIR/crosscond.csv (one row per variant, descriptor, classifier, train and test condition)
and prints the summaries. The diagonal (train = test) is taken from the time-block
cross-validation in SCORES.csv, so that it does not reuse training images."""
import os, sys, warnings
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import f1_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from classify import make, DESCS

PAIRS = [('Zoom', 'C1', 'C2'), ('Zoom', 'C3', 'C4'),
         ('Color/gray', 'C1', 'C3'), ('Color/gray', 'C2', 'C4'), ('Color/gray', 'C5', 'C6'),
         ('Chamber/natural', 'C1', 'C5'), ('Chamber/natural', 'C3', 'C6'),
         ('Indoor/outdoor', 'C5', 'C7'), ('Chamber light and arrangement', 'C1', 'C8')]


def run(variant, desc, clf, tr, X, y, cond):
    warnings.filterwarnings('ignore')
    i = cond == tr
    m = make(clf, 0).fit(X[i], y[i]); rows = []
    for te in np.unique(cond):
        if te == tr: continue
        j = cond == te; p = m.predict(X[j])
        rows.append(dict(variant=variant, desc=desc, clf=clf, train=tr, test=te,
                         f1=f1_score(y[j], p, average='macro'), mae=np.abs(p - y[j]).mean(),
                         acc1=(np.abs(p - y[j]) <= 1).mean()))
    return rows


if __name__ == '__main__':
    man, feat, scores, out = sys.argv[1:5]
    d = pd.read_csv(man); F = np.load(feat)
    y = d.level.values - 10; cond = d.cond.values
    tasks = [(v, ds, c, tr, F[f'{v}_{ds}'], y, cond) for v in ['orig', 'norm'] for ds in DESCS
             for c in ['MLP', 'RF', 'XGB'] for tr in np.unique(cond)]
    res = Parallel(n_jobs=10)(delayed(run)(*t) for t in tasks)
    x = pd.DataFrame([r for a in res for r in a])
    s = pd.read_csv(scores); s = s[(s.proto == 'block') & (s.desc != 'EXIF')]
    diag = s.groupby(['variant', 'desc', 'clf', 'cond'])[['f1', 'mae', 'acc1']].mean().reset_index()
    diag = diag.rename(columns={'cond': 'train'}).assign(test=lambda t: t.train)
    x = pd.concat([x, diag[x.columns]], ignore_index=True)
    x.to_csv(os.path.join(out, 'crosscond.csv'), index=False)
    pd.set_option('display.width', 220)
    o = x[x.variant == 'orig']; off = o[o.train != o.test]; on = o[o.train == o.test]
    print('=== Mean macro F1 (orig), within condition (time-block) vs across conditions, by descriptor')
    t = pd.DataFrame({'within': on.groupby('desc').f1.mean(), 'across': off.groupby('desc').f1.mean(),
                      'across_norm': x[(x.variant == 'norm') & (x.train != x.test)].groupby('desc').f1.mean()})
    print(t.round(2)); print('overall', t.mean().round(3).to_dict())
    print('\n=== 8x8 matrix, mean over descriptors and classifiers (rows: train, cols: test), orig')
    print(o.groupby(['train', 'test']).f1.mean().unstack().round(2))
    print('\n=== Pairs differing mainly in one factor (both directions, mean over classifiers), orig')
    rows = []
    for fac, a, b in PAIRS:
        pr = off[((off.train == a) & (off.test == b)) | ((off.train == b) & (off.test == a))]
        r = pr.groupby('desc').f1.mean(); r['factor'] = fac; r['pair'] = f'G0{a[1]}-G0{b[1]}'
        r['mean'] = pr.f1.mean(); r['within'] = on[on.train.isin([a, b])].f1.mean(); rows.append(r)
    print(pd.DataFrame(rows).set_index(['factor', 'pair'])[['within', 'mean'] + DESCS].round(2))
    print('\n=== Chamber color conditions G01, G02, G08 and their transfer (orig, mean over descriptors and classifiers)')
    ch = off[off.train.isin(['C1', 'C8']) & off.test.isin(['C1', 'C8'])]
    print('G01<->G08', round(ch.f1.mean(), 3))
    print('\n=== Across-condition +-1 accuracy and MAE (orig)', off[['acc1', 'mae']].mean().round(3).to_dict(),
          ' within:', on[['acc1', 'mae']].mean().round(3).to_dict())
    print('share of across-condition runs with F1 < 0.2:', round((off.f1 < 0.2).mean(), 2), ' > 0.5:', round((off.f1 > 0.5).mean(), 2))
