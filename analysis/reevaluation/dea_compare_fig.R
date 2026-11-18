# Proxy-based versus measured-cost CRS efficiency of the 21 re-evaluated pipelines (time-block outputs).
# Usage: Rscript dea_compare_fig.R dea_compare.csv out.eps
args <- commandArgs(trailingOnly=TRUE)
suppressMessages({library(ggplot2); library(ggrepel)})
d <- read.csv(args[1])
d <- d[d$proto == "block", ]
d$clf <- factor(d$clf, levels=c("MLP","RF","XGB"), labels=c("MLP","RF","XGBoost"))
d$desc <- factor(d$desc, levels=c("CMMS","HRLK","GLCM","LBP","G_H","RGB","HSV"))
d$lab <- paste(d$clf, d$desc, sep="-")
rho <- cor(d$CRS_proxy, d$CRS_meas, method="spearman")
set.seed(1)
p <- ggplot(d, aes(x=CRS_proxy, y=CRS_meas)) +
  geom_abline(slope=1, intercept=0, linetype="dashed", colour="grey55") +
  geom_point(aes(fill=desc, shape=clf), size=3.6, colour="black", stroke=0.4, alpha=0.9) +
  geom_text_repel(aes(label=lab), size=2.7, box.padding=0.45, point.padding=0.3,
                  min.segment.length=0, segment.colour="grey50", seed=3, max.overlaps=50) +
  annotate("text", x=0.98, y=0.04, hjust=1, size=3.6,
           label=sprintf("Spearman correlation = %.2f", rho)) +
  scale_x_continuous(limits=c(0,1.02), breaks=seq(0,1,0.2)) +
  scale_y_continuous(limits=c(0,1.02), breaks=seq(0,1,0.2)) +
  scale_shape_manual(values=c(21,22,24), name="Classifier") +
  scale_fill_brewer(palette="Dark2", name="Descriptor") +
  labs(x=expression(theta[CRS]~"with proxy input (feature length)"),
       y=expression(theta[CRS]~"with measured inputs (extraction time, model size)")) +
  coord_equal() +
  theme_bw() + theme(panel.grid.minor=element_blank(), axis.text=element_text(colour="black", size=10)) +
  guides(fill=guide_legend(override.aes=list(shape=21, size=4)),
         shape=guide_legend(override.aes=list(size=4, fill="grey70")))
ggsave(args[2], p, width=7.5, height=5.6, device=cairo_ps)
