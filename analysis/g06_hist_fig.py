"""Figure 5: 44-bin grayscale histograms of the 222 G06 images, ordered by level and file name.
Usage: python3 -I g06_hist_fig.py PNG_DIR OUT.eps   (PNGs named <level>_<file>.png, resized to 454x605)"""
import sys, os, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt, seaborn as sns
from PIL import Image
d, out = sys.argv[1:3]
files = sorted(os.listdir(d), key=lambda f: (int(f.split('_')[0]), f.split('_', 1)[1]))
H, lev, edited = [], [], []
for f in files:
    rgb = np.asarray(Image.open(os.path.join(d, f)).convert('RGB')).astype(float)
    g = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).round().clip(0, 255)
    h, _ = np.histogram(g, bins=44, range=(0, 256)); H.append(h / h.sum())
    lev.append(int(f.split('_')[0])); edited.append('edited' in f)
H = np.array(H).T; lev = np.array(lev)
fig, ax = plt.subplots(figsize=(10, 4.6))
sns.heatmap(H, cmap='rocket', ax=ax, cbar_kws={'label': 'Fraction of pixels'}, xticklabels=20, yticklabels=4)
for b in np.where(np.diff(lev))[0] + 1: ax.axvline(b, color='cyan', lw=2)
for L in np.unique(lev):
    i = np.where(lev == L)[0]; ax.text(i.mean() + 0.5, 1.5, f'H={L}%', color='white', ha='center', va='top', fontsize=10)
e = np.where(edited)[0]

ax.set_xlabel(f'Image index (n = {len(lev)}, ordered by moisture level and file name)'); ax.set_ylabel('Gray-level bin (0 = black, 43 = white)')

fig.tight_layout(); fig.savefig(out, format='eps'); fig.savefig(out.replace('.eps', '.png'), dpi=80)
print(H.shape, {int(L): int((lev == L).sum()) for L in np.unique(lev)}, 'edited:', int(np.sum(edited)))
