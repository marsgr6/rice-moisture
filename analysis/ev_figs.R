args <- commandArgs(trailingOnly=TRUE)
suppressMessages({library(ggplot2); library(plyr)})
dt <- read.csv(args[1], sep=",", header=TRUE, fileEncoding="UTF-8-BOM")
colnames(dt) <- c("group","precision","recall","f1score","support","model","data","scale","algorithm")
dt <- dt[!is.na(dt$scale) & dt$group=="macro avg", ]
dt$cat <- ifelse(dt$algorithm %in% c("CMMS","G_H","HSV","RGB"), "COLOR", "TEXTURE")
dt$GROUP <- gsub("C0", "G0", dt$data)
dt_G <- ddply(dt, c("GROUP","algorithm","cat","scale"), summarise, n=length(f1score),
              f1=mean(f1score, na.rm=TRUE), f1.sd=sd(f1score, na.rm=TRUE))
stopifnot(all(dt_G$n==3))
mk <- function(cat, shapes, title, out) {
  pd <- position_dodge(.3)
  p <- ggplot(dt_G[dt_G$cat==cat,], aes(x=as.factor(scale), y=f1, color=algorithm, group=algorithm)) +
    geom_errorbar(aes(ymin=f1-f1.sd, ymax=f1+f1.sd), width=.2, linewidth=0.25, position=pd) +
    geom_line(position=pd) + geom_point(aes(shape=algorithm), position=pd, size=2) +
    scale_shape_manual(values=shapes) +
    scale_y_continuous(breaks=seq(0,1,0.2), limits=c(0,1), oob=scales::squish) +
    scale_color_brewer(palette="Dark2") +
    geom_hline(yintercept=0.7, linetype="dashed", color="orange") +
    theme_bw() + facet_wrap(vars(GROUP), ncol=4) +
    theme(legend.position="top", legend.title=element_blank(),
          axis.line=element_line(linewidth=1, colour="black"),
          strip.background=element_rect(fill="white", colour="black", linewidth=0),
          panel.grid.major=element_line(colour="#d3d3d3"), panel.grid.minor=element_blank(),
          panel.border=element_blank(), panel.background=element_blank(),
          axis.text.x=element_text(angle=45, vjust=1, hjust=1, colour="black", size=9),
          axis.text.y=element_text(colour="black", size=9)) +
    labs(x="Scale (%)", y="F1-score", subtitle=title)
  ggsave(out, p, width=10, height=6, device=cairo_ps)
}
mk("COLOR", c(0,15,17,19), "Color features (held-out test set)", file.path(args[2], "global_color.eps"))
mk("TEXTURE", c(15,17,19,0), "Texture features (held-out test set)", file.path(args[2], "global_texture.eps"))
print(range(dt_G$f1))
