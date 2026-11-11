"""Usage: python3 -I classify.py MANIFEST FEATURES.npz OUTDIR
Protocol 'random': the paper's protocol, stratified 175/35 split, repeated 20 times.
Protocol 'block' : leave-one-time-block-out (6 consecutive-capture blocks per level).
"""
import os, sys, time, pickle, warnings
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedShuffleSplit, LeaveOneGroupOut
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score, accuracy_score
from xgboost import XGBClassifier

DESCS = ['CMMS', 'G_H', 'RGB', 'HSV', 'LBP', 'GLCM', 'HRLK']


def make(clf, seed):
    if clf == 'MLP':
        return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(100,), max_iter=200, random_state=seed))
    if clf == 'RF':
        return RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1)
    return XGBClassifier(n_estimators=100, learning_rate=0.3, max_depth=6, reg_lambda=1, random_state=seed,
                         n_jobs=1, tree_method='hist', use_label_encoder=False, eval_metric='mlogloss')


def run(cond, desc, variant, X, y, groups, idx):
    warnings.filterwarnings('ignore')
    rows, preds = [], []
    splits = [('random', k, tr, te) for k, (tr, te) in
              enumerate(StratifiedShuffleSplit(n_splits=20, test_size=35, random_state=0).split(X, y))]
    splits += [('block', k, tr, te) for k, (tr, te) in enumerate(LeaveOneGroupOut().split(X, y, groups))]
    for proto, k, tr, te in splits:
        for clf in ['MLP', 'RF', 'XGB']:
            m = make(clf, k)
            t0 = time.perf_counter(); m.fit(X[tr], y[tr]); tfit = time.perf_counter() - t0
            t0 = time.perf_counter(); p = m.predict(X[te]); tpred = (time.perf_counter() - t0) / len(te)
            err = np.abs(p - y[te])
            rows.append(dict(cond=cond, desc=desc, variant=variant, proto=proto, fold=k, clf=clf,
                             f1=f1_score(y[te], p, average='macro'), acc=accuracy_score(y[te], p),
                             mae=err.mean(), acc1=(err <= 1).mean(), n_test=len(te),
                             t_fit=tfit, t_pred=tpred, size=len(pickle.dumps(m))))
            preds += [dict(cond=cond, desc=desc, variant=variant, proto=proto, fold=k, clf=clf,
                           img=int(idx[i]), y=int(y[i]), p=int(pi)) for i, pi in zip(te, p)]
    return rows, preds


if __name__ == '__main__':
    man, feat, outdir = sys.argv[1:4]
    d = pd.read_csv(man); F = np.load(feat)
    meta = np.c_[d.brightness, np.log(d.iso), np.log(d.exposure), d.fnumber]
    tasks = []
    for cond in sorted(d.cond.unique()):
        idx = np.where(d.cond == cond)[0]
        y = d.level.values[idx] - 10; g = d.block.values[idx]
        for variant in ['orig', 'norm']:
            for desc in DESCS:
                tasks.append((cond, desc, variant, F[f'{variant}_{desc}'][idx], y, g, idx))
        tasks.append((cond, 'EXIF', 'orig', meta[idx], y, g, idx))
    res = Parallel(n_jobs=10, verbose=5)(delayed(run)(*t) for t in tasks)
    pd.DataFrame([r for a, _ in res for r in a]).to_csv(os.path.join(outdir, 'scores.csv'), index=False)
    pd.DataFrame([r for _, b in res for r in b]).to_csv(os.path.join(outdir, 'predictions.csv.gz'), index=False)
    print('done', len(res))
