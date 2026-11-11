# Usage: Rscript local_fig.R local_new.csv OUT.eps   (Fig. 6: local descriptors, 15% scale)
suppressMessages({library(ggplot2); library(plyr)})
a <- commandArgs(TRUE); dt <- read.csv(a[1])
colnames(dt) <- c("group", "precision", "recall", "f1score", "support", "model", "data")
cls <- dt[dt$group %in% as.character(10:16), ]
d3 <- ddply(dt, c("data", "model"), summarise, f1 = f1score[group == "macro avg"])
sdv <- ddply(cls, c("data", "model"), summarise, f1.sd = sd(f1score))
d3 <- merge(d3, sdv)
p <- ggplot(d3, aes(x = data, y = f1, fill = model)) +
  geom_bar(stat = "identity", width = 0.8) + facet_grid(. ~ model) +
  geom_errorbar(aes(ymin = pmax(0, f1 - f1.sd), ymax = pmin(1.03, f1 + f1.sd)), width = 0.25) +
  geom_hline(yintercept = 0.7, linetype = "dashed", color = "orange") +
  scale_y_continuous(breaks = seq(0, 1, 0.2), limits = c(0, 1.03)) + scale_fill_brewer(palette = "Dark2") +
  theme_bw() + theme(legend.position = "none", strip.background = element_rect(fill = "white"),
    panel.grid.minor = element_blank(), axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1, colour = "black", size = 9),
    axis.text.y = element_text(colour = "black", size = 9)) +
  labs(x = "Datasets", y = "F1-score")
ggsave(a[2], p, width = 8, height = 3.6, device = cairo_ps); ggsave(sub(".eps", ".png", a[2]), p, width = 8, height = 3.6, dpi = 110)
print(d3)
