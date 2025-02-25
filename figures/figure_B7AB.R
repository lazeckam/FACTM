library(ggplot2)
library(dplyr)
library(tidyverse)
library(gridExtra)
library(RColorBrewer)
library(wesanderson)

# colors
colors_pal = brewer.pal(n = 8, name = "Set2")

# data

df <- read.csv('data//mirex//mirex_weights_rotated.csv', row.names=1)
# delete 'lowlevel.' and 'tonal.' for essentia
df$feature_name <- gsub('lowlevel.', '', df$feature_name)
df$feature_name <- gsub('tonal.', '', df$feature_name)
# delete package name
df$feature_name <- gsub('_pyAA', '', df$feature_name)
df$feature_name <- gsub('_essentia', '', df$feature_name)
num_top_weights <- 15

num_views <- 4
df$view <- paste('View', df$view+1)
df$view <- factor(df$view, levels=paste('View', 1:4))
view_color = c("View 1" = colors_pal[1], "View 2" = colors_pal[2], "View 3" = colors_pal[3], 'View 4' = colors_pal[4])

df <- df

# Fig. B7 - A - rotated factor 3 ----

dfA <- df[df$factor == 2,]

tres_up <- sort((dfA$weight))[nrow(dfA)-num_top_weights]
tres_down <- sort((dfA$weight))[num_top_weights]

df_up <- dfA[dfA$weight > tres_up,]
df_up$view <- factor(df_up$view, levels=paste('View', 1:4))
df_down <- df[df$weight <= tres_down,]


fig_up <- ggplot(df_up, aes(y=weight, x=fct_reorder(feature_name, weight), fill=view)) +
  geom_bar(stat="identity", show.legend=TRUE) +
  scale_fill_manual(name = 'Views', values = view_color, drop = FALSE) + 
  scale_x_discrete(position = "top") +
  scale_y_continuous(labels = function(x0) ifelse(x0 == 0, "0.0   ", x0)) +
  coord_flip() +
  theme_minimal() +
  theme(legend.position="none") +
  theme(axis.title.x=element_blank(), axis.title.y=element_blank(), plot.margin = unit(c(0.1, 0.5, 0, 0), "cm"))

fig_down <- ggplot(df_down, aes(y=weight, x=fct_reorder(feature_name, weight,.desc = TRUE), fill=factor(view))) +
  geom_bar(stat="identity", show.legend=TRUE) +
  scale_fill_manual(name = 'Views', values = view_color, drop = FALSE) +
  coord_flip() +
  scale_y_continuous(labels = function(x0) ifelse(x0 == 0, " ", x0)) +
  theme_minimal() +
  theme(legend.position="none") +
  theme(axis.title.x=element_blank(), axis.title.y=element_blank(), plot.margin = unit(c(0.1, 0 , 0, 0.5), "cm"))

fig_dummy <- ggplot(df_up, aes(y=weight, x=fct_reorder(feature_name, weight), fill=view)) +
  geom_bar(stat="identity", show.legend=TRUE) +
  scale_fill_manual(name = 'Views', values = view_color, drop = FALSE) + 
  scale_x_discrete(position = "top") +
  coord_flip() +
  theme_minimal() +
  theme(legend.position="bottom")

fig_A <- grid.arrange(fig_down, fig_up,ncol=2, widths=c(50/100, 50/100), top='Loadings of rotated factor 3')

# Fig. B7 - B - rotated factor 6 ----

dfB <- df[df$factor == 5,]

tres_up <- sort((dfB$weight))[nrow(dfB)-num_top_weights]
tres_down <- sort((dfB$weight))[num_top_weights]

df_up <- dfB[dfB$weight > tres_up,]
df_up$view <- factor(df_up$view, levels=paste('View', 1:4))
df_down <- dfB[dfB$weight <= tres_down,]

fig_up_B <- ggplot(df_up, aes(y=weight, x=fct_reorder(feature_name, weight), fill=view)) +
  geom_bar(stat="identity", show.legend=TRUE) +
  scale_fill_manual(name = 'Views', values = view_color, drop = FALSE) + 
  scale_x_discrete(position = "top") +
  scale_y_continuous(labels = function(x0) ifelse(x0 == 0, "0.0   ", x0)) +
  coord_flip() +
  theme_minimal() +
  theme(legend.position="none") +
  theme(axis.title.x=element_blank(), axis.title.y=element_blank(), plot.margin = unit(c(0.1, 0.5 , 0, 0), "cm"))

fig_down_B <- ggplot(df_down, aes(y=weight, x=fct_reorder(feature_name, weight,.desc = TRUE), fill=factor(view))) +
  geom_bar(stat="identity", show.legend=TRUE) +
  scale_fill_manual(name = 'Views', values = view_color, drop = FALSE) +
  coord_flip() +
  scale_y_continuous(labels = function(x0) ifelse(x0 == 0, " ", x0)) +
  theme_minimal() +
  theme(legend.position="none") +
  theme(axis.title.x=element_blank(), axis.title.y=element_blank(),plot.margin = unit(c(0.1, 0 , 0, 0.5), "cm"))


fig_B <- grid.arrange(fig_down_B, fig_up_B, ncol=2, widths=c(48/100, 52/100), top='Loadings of rotated factor 6')

# Figure AB ----

g_legend <- function(a.gplot){
  tmp <- ggplot_gtable(ggplot_build(a.gplot))
  leg <- which(sapply(tmp$grobs, function(x) x$name) == "guide-box")
  legend <- tmp$grobs[[leg]]
  return(legend)
  }
mylegend <- g_legend(fig_dummy)

fig_AB <- grid.arrange(arrangeGrob(fig_A,
                                   fig_B ,nrow=1),
                       mylegend, nrow=2,heights=c(10, 1))
fig_AB
