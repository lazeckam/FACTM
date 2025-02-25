library(ggplot2)
library(plyr)
library(dplyr)
library(tidyverse)
library(tidyr)
library(reshape2)
library(grid)
library(ggpubr)
library(rstatix)
library(gridExtra)
library(ggplot2)
library(rstatix)

# Figure 5A ----

df_pos_neg <- read.csv('data//mirex//mirex_topics_sentiment.csv', row.names = 1)
mean_pos <- mean(df_pos_neg$positive)
mean_neg <- mean(df_pos_neg$negative)
df_pos_neg <- pivot_longer(df_pos_neg, cols=c('positive', 'negative'))
df_pos_neg$topic <- factor(as.numeric(gsub("Topic ", '', df_pos_neg$topic)))
df_pos_neg$name <- factor(df_pos_neg$name, levels = c('positive', 'negative'))

fig_A <- ggplot(df_pos_neg, aes(x=topic, y=value, fill=name, width=.75))+
  geom_bar(stat="identity", position=position_dodge()) +
  scale_fill_manual(values=c('green3', 'red3'), name='Sentiment') +
  theme_bw() +
  geom_hline(yintercept = mean_pos, color='green3', linetype='dashed') +
  geom_hline(yintercept = mean_neg, color='red3', linetype='dashed') +
  theme(legend.position=c(0.82,0.8), legend.title = element_text( size=7), legend.text=element_text(size=7),
        legend.key.size = unit(0.25, "cm")) +
  xlab('Topic') + ylab('Weighted fraction of words')
fig_A

# Figure 5B ----

eps <- 0.5
symnum.args <- list(cutpoints = c(0, 0.001, 0.01, 0.05, Inf), symbols = c("***", "**", "*", "ns"))

df <- read.csv('data//mirex//mirex_eta_mu_class.csv', row.names = 1)
df_tmp <- df[df$topic == 'Topic 3',]

stat.test <- df_tmp %>%
  pairwise_wilcox_test(eta~class, p.adjust.method = "bonferroni", symnum.args=symnum.args)
stat.test$p.adj.signif <- cut(stat.test$p.adj, symnum.args$'cutpoints', symnum.args$'symbols')
stat.test <- stat.test %>% add_xy_position(x = "class")
how_many_significant <- sum(stat.test$p.adj < 0.05)
stat.test$y.position <- rep(NA, 10)
stat.test$y.position[stat.test$p.adj < 0.05] <- seq(max(df_tmp$eta)+eps, max(df_tmp$eta)+how_many_significant/1.95, length=how_many_significant)

fig_B <- ggboxplot(df_tmp, x = "class", y = "eta", fill='lightgray') + 
  stat_pvalue_manual(stat.test, label = "p.adj.signif", hide.ns = TRUE) +
  ggtitle('Topic 3') +
  theme_bw() +
  ylab(expression(eta[''%.%' 3'])) +
  xlab('Class')
fig_B

# Figure 5C ----

df_tmp <- df[df$topic == 'Topic 4',]

stat.test <- df_tmp %>% pairwise_wilcox_test(eta~class, p.adjust.method = "bonferroni")
stat.test$p.adj.signif <- cut(stat.test$p.adj, symnum.args$'cutpoints', symnum.args$'symbols')
stat.test <- stat.test %>% add_xy_position(x = "class")
how_many_significant <- sum(stat.test$p.adj < 0.05)
stat.test$y.position <- rep(NA, 10)
stat.test$y.position[stat.test$p.adj < 0.05] <- seq(max(df_tmp$eta)+eps, max(df_tmp$eta)+how_many_significant/2.9, length=how_many_significant)

fig_C <- ggboxplot(df_tmp, x = "class", y = "eta", fill='lightgray') + 
  stat_pvalue_manual(stat.test, label = "p.adj.signif", hide.ns = TRUE) +
  ggtitle('Topic 4') +
  theme_bw() +
  ylab(expression(eta[''%.%' 4'])) +
  xlab('Class')
fig_C

fig_BC <- grid.arrange(fig_B, fig_C, nrow = 1)#, top=textGrob("Probabilities of for samples split by classes",gp=gpar(fontsize=20)))
fig_A <- grid.arrange(fig_A, nrow = 1)#,top=textGrob("Mean word sentiment",gp=gpar(fontsize=20)))
fig_5 <- grid.arrange(fig_A, fig_BC, nrow = 1,widths=c(1,2))
fig_5


# Boxplots - function ----

one_plot_factor <- function(df, topic_name, variable_name, y_lab){
  
  df_tmp <- df[df$factor == topic_name,]
  
  
  stat.test <- df_tmp %>%
    pairwise_wilcox_test(
      as.formula(paste(variable_name, "~ class")), 
      p.adjust.method = "bonferroni", symnum.args=symnum.args
    )
  stat.test$p.adj.signif <- cut(stat.test$p.adj, symnum.args$'cutpoints', symnum.args$'symbols')
  stat.test <- stat.test %>% add_xy_position(x = "class")
  how_many_significant <- sum(stat.test$p.adj < 0.05)
  stat.test$y.position <- rep(NA, 10)
  stat.test$y.position[stat.test$p.adj < 0.05] <- seq(max(df_tmp[variable_name])+eps, max(df_tmp[variable_name])+how_many_significant/3, length=how_many_significant)
  
  
  fig <- ggboxplot(data=df_tmp, x = 'class', y = variable_name, fill='fill_var') + 
    scale_fill_manual(values=c('lightgrey', 'red3')) +
    ggtitle(topic_name) +
    ylab(y_lab) + 
    stat_pvalue_manual(data=stat.test, label = "p.adj.signif", hide.ns = TRUE) +
    xlab('Class') +
    theme_bw() +
    guides(fill="none")
  return(fig)
}

# Figure B6 A ---- 

df <- read.csv('data//mirex//mirex_factors_classes.csv', row.names = 1)
df$fill_var <- 'A'

list_of_plots <- list()

for( i in 1:10){
  if (i == 1 | i== 6){
    list_of_plots[[i]] <- one_plot_factor(df, paste0('Factor ', i), 'value', 'Factor values')
  }else{
    list_of_plots[[i]] <- one_plot_factor(df, paste0('Factor ', i), 'value', '')
  }
}

fig_A <- do.call(grid.arrange, c(list_of_plots, nrow=2, top=' '))
fig_A

# Figure B6 B ---- 

df <- read.csv('data//mirex//mirex_rotated_factors_classes.csv', row.names = 1)

df$fill_var <- 'A'
for(i in 1:5){
  df[df$factor==paste0('Factor ', i) & df$class==i, 'fill_var'] <- 'B'
}

change_fac_names <- paste0('Rotated factor ', 1:10)
names(change_fac_names) <- paste0('Factor ', 1:10)

df$factor <- revalue(df$factor, change_fac_names )


list_of_plots <- list()

for( i in 1:10){
  if (i == 1 | i== 6){
    list_of_plots[[i]] <- one_plot_factor(df, paste0('Rotated factor ', i), 'value', 'Factor values')
  }else{
    list_of_plots[[i]] <- one_plot_factor(df, paste0('Rotated factor ', i), 'value', '')
  }
}

fig_B <- do.call(grid.arrange, c(list_of_plots, nrow=2, top=' '))
fig_B


fig_B6 <- grid.arrange(fig_A, fig_B, nrow = 2)
fig_B6
