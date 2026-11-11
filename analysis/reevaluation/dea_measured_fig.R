# Usage: Rscript dea_measured_fig.R dea_measured_block.csv out.eps
args <- commandArgs(trailingOnly=TRUE)
suppressMessages({library(ggplot2); library(ggrepel)})
d <- read.csv(args[1])
d$clf <- factor(d$clf, levels=c("MLP","RF","XGB"), labels=c("MLP","RF","XGBoost"))
d$desc <- gsub("_", "_", d$desc)
d$CRS <- pmin(pmax(d$CRS, 0), 1)
d$lab <- paste(d$clf, d$desc, sep="-")
d$eff <- d$CRS > 0.999
d <- d[order(-d$size_kB), ]  # draw small models on top
set.seed(1)
p <- ggplot(d, aes(x=ext_ms, y=F1)) +
  geom_point(aes(fill=CRS, shape=clf, size=size_kB), colour="black", stroke=0.4, alpha=0.9) +
  geom_text_repel(data=d[d$eff,], aes(label=lab), size=3, box.padding=0.6, point.padding=0.4,
                  min.segment.length=0, segment.colour="grey40", seed=3, max.overlaps=50, nudge_x=0.12, direction="y", hjust=0) +
  scale_x_log10(breaks=c(3,10,30,100,300)) +
  scale_shape_manual(values=c(21,22,24), name="Classifier") +
  scale_fill_viridis_c(name=expression(theta[CRS]), limits=c(0,1), oob=scales::squish) +
  scale_size_continuous(name="Model size (kB)", range=c(2,7), breaks=c(100,500,2000)) +
  labs(x="Extraction time per image (ms, logarithmic scale)", y="Mean macro F1-score (time-block validation)") +
  theme_bw() + theme(panel.grid.minor=element_blank(), axis.text=element_text(colour="black", size=10)) +
  guides(shape=guide_legend(override.aes=list(size=4, fill="grey70")))
ggsave(args[2], p, width=8, height=5.2, device=cairo_ps)
