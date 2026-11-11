# Rice grain moisture classification from smartphone images: code and results

Code and results for the paper *Rice Moisture Classification from Smartphone Images: Balancing Performance, Stability, and Cost with Data Envelopment Analysis* (manuscript sustainability-4666494).

Seven moisture levels (10–16%) of INIAP 11 rice grains were photographed with an iPhone 13 Pro under eight acquisition conditions (G01–G08; in the code, `C1`–`C8`). Global and local image descriptors were classified with MLP, Random Forest, and XGBoost, and the global-descriptor pipelines were compared with Data Envelopment Analysis (DEA).

The photographs (about 3.3 GB) are **not included** because of their size. They are available from the corresponding author upon reasonable request.

## Contents

| Path | Content |
|---|---|
| `0.0. Check number.ipynb` | Number of training and test images per condition and moisture level |
| `0.1. Group visualization .ipynb` | Example images of the eight acquisition conditions |
| `1.1. <descriptor> (Gxx).ipynb` | One notebook per descriptor and condition: feature extraction, 10-fold cross-validation on the training images, and evaluation on the held-out test images |
| `RESULTS/GLOBAL/CV/<scale>/` | 10-fold cross-validation results of the global descriptors (one CSV per condition, descriptor, and classifier; scales 3, 5, 10, 15, 20, and 25%) |
| `RESULTS/GLOBAL/EV/` | Test-set classification reports of the global descriptors (per descriptor and condition), merged in `mergedEV.csv` |
| `RESULTS/LOCAL/` | Codebook-size cross-validation (`CV/`) and test-set results (`EV/`) of the local descriptors |
| `RESULTS/*.Rmd`, `RESULTS/*.html` | R notebooks that summarize the results and draw the figures, with their rendered output |
| `analysis/` | Scripts for the DEA and for the figures of the revised manuscript (see below) |

### Descriptor notebooks

Global descriptors (computed at scales of 3, 5, 10, 15, 20, and 25% of the original resolution):
color moments (`Color Moments`), grayscale histogram (`Gray histogram`), RGB histogram, HSV histogram, local binary patterns (`LBP`), gray-level co-occurrence matrix properties (`GLCM`), and Haralick features (`Haralick`).

Local descriptors (computed at the 15% scale and encoded with a bag of visual words): `AKAZE`, `BRISK`, `ORB`, and `SIFT`.

Each notebook processes one scale per run (variable `scale`) and writes its results to `RESULTS/`.

### Analysis scripts (`analysis/`)

| Script | Purpose |
|---|---|
| `dea_main.py` | CCR/BCC DEA of the 126 global pipelines with slack and sensitivity analyses (Tables 8–12): `python3 analysis/dea_main.py RESULTS/GLOBAL/EV/mergedEV.csv out/` |
| `dea_figs.R` | Figures 8 and 9 (DEA): `Rscript analysis/dea_figs.R out/dea_results.csv fig8.eps fig9.eps` |
| `ev_figs.R` | Figures 3 and 4 (test-set $F_1$ of the global descriptors by scale): `Rscript analysis/ev_figs.R RESULTS/GLOBAL/EV/mergedEV.csv` |
| `local_fig.R` | Figure 6 (local descriptors): `Rscript analysis/local_fig.R RESULTS/LOCAL/EV/local_new.csv fig6.eps` |
| `heatmap_fig.py` | Figure 7 (all descriptors at 15%, Random Forest): `python3 analysis/heatmap_fig.py RESULTS/GLOBAL/EV/mergedEV.csv RESULTS/LOCAL/EV/local_new.csv fig7.eps` |
| `g06_hist_fig.py` | Figure 5 (grayscale histograms of G06); requires the photographs resized to PNG |

## Running the notebooks

The notebooks expect the photographs in the working directory with this layout:

```
DATA/TRAINING/C1/<level>/...   DATA/TEST/C1/<level>/...    (C1 ... C8)
```

and write intermediate features to `FEATURES/` and local-descriptor models to `models/` (both excluded from the repository). Five images per moisture level and condition were randomly assigned to `DATA/TEST`; the rest form `DATA/TRAINING`.

## Requirements

Python 3 with the packages in `requirements.txt` (HEIC photographs are read with `pillow-heif`), and R with `ggplot2`, `plyr`, and `knitr` for the R notebooks and figure scripts.

## Citation

If you use this code, please cite the paper (reference to be added upon publication).
