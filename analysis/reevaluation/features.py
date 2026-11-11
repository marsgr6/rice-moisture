"""Re-implementation of the 7 global descriptors of the paper (scale 15%).
Usage: python3 -I features.py MANIFEST IMGDIR OUT.npz
Produces, for each image, features for the original image ('orig') and for an
intensity-normalized image ('norm': each RGB channel standardized per image,
clipped to +-3 SD and mapped back to 0..255), plus per-descriptor timings.
"""
import os, sys, time
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np, pandas as pd
from multiprocessing import Pool
from PIL import Image
from scipy.stats import skew
from skimage.color import rgb2hsv
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
DESCS = ['CMMS', 'G_H', 'RGB', 'HSV', 'LBP', 'GLCM', 'HRLK']


def to_gray(rgb):
    return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).round().clip(0, 255).astype(np.uint8)


def cmms(rgb):
    hsv = rgb2hsv(rgb).reshape(-1, 3)
    return np.concatenate([hsv.mean(0), hsv.std(0), skew(hsv, axis=0)])          # 9


def g_h(gray):
    h, _ = np.histogram(gray, bins=44, range=(0, 256)); return h / h.sum()        # 44


def rgb_hist(rgb):
    return np.concatenate([np.bincount(rgb[..., c].ravel(), minlength=256) for c in range(3)]) / rgb[..., 0].size  # 768


def hsv_hist(rgb):
    hsv = (rgb2hsv(rgb) * 255).round().clip(0, 255).astype(np.uint8)
    return np.concatenate([np.bincount(hsv[..., c].ravel(), minlength=256) for c in range(3)]) / rgb[..., 0].size  # 768


def lbp(gray):
    P, R = 30, 3                                                                    # uniform LBP -> P+2 = 32 bins
    code = local_binary_pattern(gray, P, R, method='uniform')
    h = np.bincount(code.astype(int).ravel(), minlength=P + 2)[:P + 2]; return h / h.sum()


def glcm_feats(P):
    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
    return np.concatenate([graycoprops(P, p).ravel() for p in props])               # 5 x 4 = 20


def haralick13(p):
    """13 Haralick features of one normalized symmetric GLCM p (L x L)."""
    eps = 1e-12
    L = p.shape[0]; i, j = np.indices((L, L))
    px, py = p.sum(1), p.sum(0); lv = np.arange(L)
    mux, muy = (lv * px).sum(), (lv * py).sum()
    sx, sy = np.sqrt(((lv - mux) ** 2 * px).sum()), np.sqrt(((lv - muy) ** 2 * py).sum())
    f1 = (p ** 2).sum()
    pxmy = np.bincount((np.abs(i - j)).ravel(), weights=p.ravel(), minlength=L)
    f2 = (np.arange(L) ** 2 * pxmy).sum()
    f3 = ((i * j * p).sum() - mux * muy) / (sx * sy + eps)
    f4 = ((i - mux) ** 2 * p).sum()
    f5 = (p / (1 + (i - j) ** 2)).sum()
    pxpy = np.bincount((i + j).ravel(), weights=p.ravel(), minlength=2 * L - 1); k = np.arange(2 * L - 1)
    f6 = (k * pxpy).sum()
    f8 = -(pxpy * np.log(pxpy + eps)).sum()
    f7 = ((k - f6) ** 2 * pxpy).sum()
    f9 = -(p * np.log(p + eps)).sum()
    f10 = ((np.arange(L) - (np.arange(L) * pxmy).sum()) ** 2 * pxmy).sum()
    f11 = -(pxmy * np.log(pxmy + eps)).sum()
    hx, hy = -(px * np.log(px + eps)).sum(), -(py * np.log(py + eps)).sum()
    pp = np.outer(px, py)
    hxy1 = -(p * np.log(pp + eps)).sum(); hxy2 = -(pp * np.log(pp + eps)).sum()
    f12 = (f9 - hxy1) / (max(hx, hy) + eps)
    f13 = np.sqrt(max(0.0, 1 - np.exp(-2 * (hxy2 - f9))))
    return np.array([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13])


def describe(rgb):
    out, tim = {}, {}
    t0 = time.perf_counter(); gray = to_gray(rgb); tg = time.perf_counter() - t0
    for name, fn, src in [('CMMS', cmms, 'rgb'), ('G_H', g_h, 'gray'), ('RGB', rgb_hist, 'rgb'),
                          ('HSV', hsv_hist, 'rgb'), ('LBP', lbp, 'gray')]:
        t0 = time.perf_counter(); out[name] = fn(rgb if src == 'rgb' else gray)
        tim[name] = time.perf_counter() - t0 + (tg if src == 'gray' else 0)
    t0 = time.perf_counter()
    P = graycomatrix(gray, [1], ANGLES, levels=256, symmetric=True, normed=True)
    tglcm = time.perf_counter() - t0
    t0 = time.perf_counter(); out['GLCM'] = glcm_feats(P); tim['GLCM'] = time.perf_counter() - t0 + tglcm + tg
    t0 = time.perf_counter(); out['HRLK'] = np.mean([haralick13(P[:, :, 0, a]) for a in range(4)], axis=0)
    tim['HRLK'] = time.perf_counter() - t0 + tglcm + tg
    return out, tim


def normalize(rgb):
    x = rgb.astype(np.float64)
    z = (x - x.mean((0, 1))) / (x.std((0, 1)) + 1e-9)
    return ((z.clip(-3, 3) + 3) / 6 * 255).round().astype(np.uint8)


def work(png):
    rgb = np.asarray(Image.open(png).convert('RGB'))
    fo, to = describe(rgb)
    fn, _ = describe(normalize(rgb))
    return fo, to, fn


if __name__ == '__main__':
    man, imgdir, out = sys.argv[1:4]
    d = pd.read_csv(man)
    pngs = [os.path.join(imgdir, f'{r.cond}_{r.level}_{r.file[:-5]}.png') for r in d.itertuples()]
    with Pool(10) as pool:
        res = pool.map(work, pngs, chunksize=4)
    arrs = {}
    for v, k in [('orig', 0), ('norm', 2)]:
        for name in DESCS:
            arrs[f'{v}_{name}'] = np.vstack([r[k][name] for r in res])
    for name in DESCS:
        arrs[f'time_{name}'] = np.array([r[1][name] for r in res])
    np.savez_compressed(out, **arrs)
    print({k: v.shape for k, v in arrs.items() if k.startswith('orig')})
