# Figures 8 and 9 of the paper (DEA of the global-descriptor pipelines).
# Usage: Rscript dea_figs.R dea_results.csv fig8.eps fig9.eps
# dea_results.csv is written by dea_main.py.
suppressMessages({library(ggplot2); library(ggrepel)})
a <- commandArgs(TRUE)
d <- read.csv(a[1])
d$model <- factor(c(MLP = "MLP", RFC = "RF", XGBC = "XGBoost")[d$model], levels = c("MLP", "RF", "XGBoost"))
ord <- c("CMMS", "LBP", "GLCM", "G_H", "HRLK", "RGB", "HSV")
lab <- setNames(paste0(ord, " (", c(9, 18, 24, 44, 52, 4096, 4096), ")"), ord)
d$descr <- factor(lab[d$algorithm], levels = lab)
d$crs <- pmin(d$crs, 1)
d$frontier <- d$crs > 1 - 1e-6
d$name <- paste0(d$model, ", ", d$algorithm, ", ", d$scale, "%")
pal <- c("#E69F00", "#009E73", "#56B4E9", "#0072B2", "#999999", "#D55E00", "#CC79A7")
th <- theme_bw() + theme(panel.grid.minor = element_blank(), legend.key.size = unit(0.4, "cm"),
                         legend.text = element_text(size = 8), legend.title = element_text(size = 9))

# Figure 8: F1 versus stability
p8 <- ggplot(d, aes(f1, stability)) +
  geom_point(aes(colour = descr, shape = model, size = crs), alpha = 0.85) +
  geom_point(data = d[d$frontier, ], aes(size = crs), shape = 21, colour = "black", stroke = 0.9) +
  scale_colour_manual(values = setNames(pal, lab), name = "Descriptor (length)") +
  scale_shape_manual(values = c(16, 15, 17), name = "Classifier") +
  scale_size(range = c(0.8, 4.5), limits = c(0, 1), breaks = c(0.25, 0.5, 0.75, 1), name = "CRS efficiency") +
  labs(x = expression("Mean macro " * F[1] * "-score"), y = expression("Acquisition-condition stability (1 - " * sigma / mu * ")")) + th
ggsave(a[2], p8, width = 7, height = 4.6, device = cairo_ps)

# Figure 9: F1 versus descriptor, ordered by feature length
pos <- setNames(1:7, ord)
set.seed(1)
d$x <- pos[d$algorithm] + runif(nrow(d), -0.18, 0.18)
p9 <- ggplot(d, aes(x, f1)) +
  geom_point(aes(fill = crs, size = stability), shape = 21, colour = "grey30", stroke = 0.2) +
  geom_point(data = d[d$frontier, ], aes(size = stability), shape = 21, colour = "black", stroke = 1) +
  geom_label_repel(data = d[d$frontier, ], aes(label = name), size = 2.5, min.segment.length = 0, label.size = 0.1,
                   box.padding = 0.4, force = 3, max.overlaps = 50, seed = 3, direction = "both") +
  scale_x_continuous(breaks = 1:7, labels = sub(" ", "\n", lab), limits = c(0.5, 7.5)) +
  scale_fill_viridis_c(limits = c(0, 1), name = "CRS\nefficiency") +
  scale_size(range = c(0.8, 4), name = "Stability") +
  labs(x = "Descriptor (feature vector length)", y = expression("Mean macro " * F[1] * "-score")) + th
ggsave(a[3], p9, width = 7, height = 4.6, device = cairo_ps)
