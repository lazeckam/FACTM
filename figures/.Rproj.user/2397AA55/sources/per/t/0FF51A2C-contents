library(plyr)
library(dplyr)
library(ggplot2)
library(tidyverse)
library(reshape)
library(grid)
library(gridExtra)
library(ggpubr)

# cell names:
cell_names = c("B cells"='B', "CD4 T cells" = 'CD4 T', "CD8 T cells" = 'CD8 T', "DC1" = "DC1"   ,                     
               "DC2" ='DC2', "Epithelial cells" = 'Epithelial', "Mast cells" = 'Mast' , "Migratory DC" = "Migratory DC",  
               "MoAM" = "MoAM", "Monocytes"="Monocytes",                 "Perivascular macrophages"="PVM", 
               "Plasma cells" = 'Plasma', "Proliferating T cells" = 'Prol. T',    "Proliferating macrophages"="Prol. macs.",
               "SARS-CoV-2"="SARS-CoV-2","TRAM" ="TRAM","Tregs"='Tregs', "gdT cells and NK cells"="gdT and NK",
               "pDC" ="pDC" )     

# Fig. A ----

df_contingency <- read.csv('data//covid//covid_contingency.csv')
cell_types_names <- df_contingency$row_0
df_contingency$row_0 <- factor(df_contingency$row_0, levels=cell_types_names)
topics_names <- as.character(as.numeric(gsub('X', '', colnames(df_contingency)[2:20])) +1)

df_contingency <- melt(df_contingency, id=c('row_0'))
df_contingency$variable <- factor(paste0('Topic ', as.character(as.numeric(gsub('X', '', df_contingency$variable)) +1)), levels=paste0('Topic ',topics_names))
df_contingency$row_0 <- factor(df_contingency$row_0, levels=cell_types_names)

df_contingency$row_0 <- revalue(df_contingency$row_0, cell_names)

fig_tmp <- ggplot(aes(x=variable, y=row_0, fill=value), data=df_contingency)
fig_A <- fig_tmp + geom_tile() + 
  scale_fill_gradient(low="white", high="royalblue3", name='frac.') +
  scale_y_discrete(limits=rev) +
  ylab('Cell type') +
  xlab(' ') +
  theme_bw() + theme(axis.text.x=element_text(angle = 45, hjust = 1),
                     plot.margin = margin(b = 18,t=5, unit = "pt"))
fig_A

# Fig. B ----

df_expression <- read.csv('data//covid//covid_expression.csv', stringsAsFactors = FALSE, check.names = FALSE)
colnames(df_expression)[1] <- 'X'
gene_names <- df_expression$X
df_expression <- melt(df_expression, id=c('X'))
df_expression$X <- factor(df_expression$X, levels=gene_names)
df_expression$variable <- factor(df_expression$variable, levels=cell_types_names)
df_expression$status <- 'Averaged expression'
df_expression$variable <- revalue(df_expression$variable, cell_names)

df_topics <- read.csv('data//covid//covid_topic.csv', stringsAsFactors = FALSE, check.names = FALSE)
colnames(df_topics)[1] <- 'X'
gene_names <- df_topics$X
df_topics <- melt(df_topics, id=c('X'))
df_topics$X <- factor(df_topics$X, levels=gene_names)
df_topics$variable <-  factor(paste0('Topic ', as.character(as.numeric(gsub('X', '', df_topics$variable)) +1)), levels=paste0('Topic ',topics_names))
df_topics$status <- 'Abundance inferred by FACTM'

df <- rbind(df_expression, df_topics)

fig_tmp <- ggplot(aes(x=variable, y=X, fill=value), data=df)
fig_B <- fig_tmp + geom_tile() + 
  scale_fill_gradient2(low="royalblue3", mid='white', high="red3", name='z-score') +
  scale_y_discrete(limits=rev) +
  ylab('Genes') +
  xlab(' ') +
  theme_bw() + theme(axis.text.x=element_text(angle = 45, hjust = 1), 
                     axis.text.y=element_blank(),
                     axis.ticks.y=element_blank())+
  facet_wrap(~status, scales='free_x') 
fig_B

p1 <- grid.arrange(fig_A, nrow = 1, top=textGrob("Matching topics to cell types",gp=gpar(fontsize=20)))
p2 <- grid.arrange(fig_B, nrow = 1,top=textGrob("Gene abundance",gp=gpar(fontsize=20)))
fig <- grid.arrange(p1, p2, nrow = 1,widths=c(1.5,2))
fig


