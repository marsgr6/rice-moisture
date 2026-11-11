"""Usage: python3 -I heatmap_fig.py mergedEV.csv local_new.csv OUT.eps
Fig. 7: macro F1 at 15% scale with Random Forest, global (mergedEV) and local (local_new) descriptors.
The local descriptors were classified with Random Forest only, so the global rows use Random Forest too."""
import sys, pandas as pd, seaborn as sns, matplotlib; matplotlib.use('Agg')
ev, loc, out = sys.argv[1:4]
m = pd.read_csv(ev, encoding='utf-8-sig').dropna(subset=['model'])
m = m[(m.group == 'macro avg') & (m.scale == 15) & (m.model == 'RFC')]
g = m.pivot_table(index='algorithm', columns='data', values='f1-score')
g.columns = ['G' + c[1:] for c in g.columns]
l = pd.read_csv(loc); l = l[l.group == 'macro avg'].pivot_table(index='model', columns='dataset', values='f1-score')
h = pd.concat([g, l]).round(2); h.index.name = 'Descriptor'; h.columns.name = 'Acquisition condition'
print(h)
cg = sns.clustermap(h, cmap='YlGnBu', vmin=0.2, vmax=1, annot=True, fmt='.2f', linewidths=.5, figsize=(7, 6.2),
                    cbar_pos=(0.02, 0.85, 0.03, 0.12), annot_kws={'size': 11}, dendrogram_ratio=(.12, .12))
cg.ax_cbar.set_title('$F_1$', fontsize=9); cg.ax_heatmap.set_ylabel('Descriptor'); cg.ax_heatmap.set_xlabel('Acquisition condition')
cg.ax_heatmap.tick_params(axis='y', rotation=0)
cg.savefig(out)
