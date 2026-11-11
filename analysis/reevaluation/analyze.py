"""Usage: python3 -I analyze.py OUTDIR FEATURES.npz MERGED_EV.csv"""
import sys, numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
out, feat, merged = sys.argv[1:4]
s = pd.read_csv(f'{out}/scores.csv')
s['setting'] = np.where(s.cond.isin(['C5', 'C6', 'C7']), 'natural', 'chamber')
D = ['CMMS', 'G_H', 'RGB', 'HSV', 'LBP', 'GLCM', 'HRLK']

def q(x): return pd.Series({'mean': x.mean(), 'lo': x.quantile(.025), 'hi': x.quantile(.975)})

# 1. classifier-averaged F1 per fold, then summary per condition x descriptor x protocol
f = s.groupby(['variant', 'proto', 'cond', 'desc', 'fold']).f1.mean().reset_index()
print('\n=== (1) Macro F1 (avg over classifiers), original images: random (20 repeats) vs time-block (6 folds) ===')
o = f[f.variant == 'orig']
t = o.groupby(['desc', 'proto', 'cond']).f1.mean().unstack('cond').round(2)
print(t)

print('\n=== (2) Per descriptor, mean over conditions, by setting ===')
o2 = s[s.variant == 'orig'].groupby(['desc', 'proto', 'setting']).f1.mean().unstack(['setting', 'proto']).round(3)
print(o2)

print('\n=== (3) Uncertainty of a single 35-image test estimate (random protocol, spread over 20 splits) ===')
r = s[(s.variant == 'orig') & (s.proto == 'random') & (s.desc != 'EXIF')]
w = r.groupby(['desc', 'cond', 'clf']).f1.agg(lambda x: x.quantile(.975) - x.quantile(.025))
print('median width of 95% range across splits:', round(w.median(), 3), ' IQR', w.quantile([.25, .75]).round(3).tolist())
sd = r.groupby(['desc', 'cond', 'clf']).f1.std(); print('median SD across splits:', round(sd.median(), 3))

print('\n=== (4) Drop random -> block (mean over conditions & classifiers) ===')
dr = s[s.variant == 'orig'].groupby(['desc', 'setting', 'proto']).f1.mean().unstack('proto')
dr['drop'] = dr['random'] - dr['block']; print(dr.round(3))
print('overall:', s[s.variant == 'orig'].groupby('proto').f1.mean().round(3).to_dict())

print('\n=== (5) Intensity-normalization ablation (block protocol) ===')
ab = s[(s.proto == 'block') & (s.desc != 'EXIF')].groupby(['desc', 'setting', 'variant']).f1.mean().unstack('variant')
ab['change'] = ab['norm'] - ab['orig']; print(ab.round(3))
ab2 = s[(s.proto == 'random') & (s.desc != 'EXIF')].groupby(['desc', 'setting', 'variant']).f1.mean().unstack('variant')
ab2['change'] = ab2['norm'] - ab2['orig']; print('random protocol:'); print(ab2.round(3))

print('\n=== (6) EXIF metadata baseline (brightness, ISO, exposure, aperture) ===')
print(s[s.desc == 'EXIF'].groupby(['cond', 'proto']).f1.mean().unstack('proto').round(2))

print('\n=== (7) Ordinal errors (block protocol, original) ===')
print(s[(s.variant == 'orig') & (s.proto == 'block')].groupby(['desc', 'setting'])[['f1', 'acc', 'mae', 'acc1']].mean().unstack('setting').round(2))
p = pd.read_csv(f'{out}/predictions.csv.gz')
for name, sel in [('HRLK chamber', (p.desc == 'HRLK') & ~p.cond.isin(['C5', 'C6', 'C7'])),
                  ('HRLK natural', (p.desc == 'HRLK') & p.cond.isin(['C5', 'C6', 'C7']))]:
    pp = p[sel & (p.variant == 'orig') & (p.proto == 'block')]
    cm = pd.crosstab(pp.y + 10, pp.p + 10, normalize='index').round(2)
    print(f'\nConfusion matrix (row-normalized), {name}, block protocol, 3 classifiers pooled:'); print(cm)
    print('share of errors that are adjacent (|d|=1):', round(((pp.y - pp.p).abs() == 1).sum() / max(1, (pp.y != pp.p).sum()), 3))

print('\n=== (8) Cost: extraction time (ms/image, 15% scale, 1 core), fit/predict time, model size ===')
F = np.load(feat)
ext = pd.Series({d: np.median(F[f'time_{d}']) * 1000 for d in D}).round(1)
c = s[(s.variant == 'orig') & (s.desc != 'EXIF')].groupby(['desc', 'clf'])[['t_fit', 't_pred', 'size']].median()
c['t_pred_ms'] = c.t_pred * 1000; c['size_kB'] = c['size'] / 1024
print('extraction ms/image:'); print(ext)
print(c[['t_fit', 't_pred_ms', 'size_kB']].unstack('clf').round(3))
best = s[(s.variant == 'orig') & (s.desc != 'EXIF')].groupby('desc').f1.mean()
print('Spearman(extraction time, feature length):', round(spearmanr(ext[D], [9, 44, 768, 768, 32, 20, 13]).correlation, 2))

print('\n=== (9) Validation of the re-implementation against mergedEV.csv (15%, random protocol) ===')
m = pd.read_csv(merged, encoding='utf-8-sig').dropna(subset=['model'])
m = m[(m.group == 'macro avg') & (m.scale == 15)]
m['clf'] = m.model.map({'MLP': 'MLP', 'RFC': 'RF', 'XGBC': 'XGB'}); m['cond'] = 'C' + m.data.str[-1]
paper = m.groupby(['algorithm', 'cond'])['f1-score'].mean()
ours = s[(s.variant == 'orig') & (s.proto == 'random')].groupby(['desc', 'cond']).f1.mean()
ours.index.names = ['algorithm', 'cond']; j = pd.concat([paper.rename('paper'), ours.rename('ours')], axis=1).dropna()
print('n=', len(j), ' Pearson r=', round(pearsonr(j.paper, j.ours)[0], 2), ' mean paper', round(j.paper.mean(), 3), ' mean ours', round(j.ours.mean(), 3))
print(j.groupby(level=0).mean().round(3))
print('share of paper single-split values inside our 95% random-split range:')
rr = o[o.proto == 'random'].groupby(['desc', 'cond']).f1.agg([lambda x: x.quantile(.025), lambda x: x.quantile(.975)])
rr.columns = ['lo', 'hi']; rr.index.names = ['algorithm', 'cond']; jj = j.join(rr)
print(round(((jj.paper >= jj.lo) & (jj.paper <= jj.hi)).mean(), 2))
