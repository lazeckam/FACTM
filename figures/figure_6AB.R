library(ggplot2)
library(dplyr)
library(tidyverse)
library(reshape2)
library(grid)
library(plyr)

# significance levels
symnum.args <- list(cutpoints = c(0, 0.001, 0.01, 0.05, Inf), symbols = c("***", "**", "*", "ns"))
levels <- symnum.args[['cutpoints']][c(2,3,4)]

# data
df_factors <- read.csv('data//mirex//mirex_pvaluesMWtest_factors_classes.csv')
colnames(df_factors)[1] <- 'Factor'
df_factors$Type <- 'Original factors'

df_factors_rotation <- read.csv('data//mirex//mirex_pvaluesMWtest_factors_rotated_classes.csv')
df_factors_rotation$Type <- 'Rotated factors'
colnames(df_factors_rotation)[1] <- 'Factor'

df_factors <- rbind(df_factors, df_factors_rotation)
df_factors$Factor <- factor(df_factors$Factor, levels=paste0('Factor ', 1:10))

plot.data <- melt(df_factors, id=c('Factor', 'Type'))
# *100 - bonferroni correction
plot.data$stars <- cut(plot.data$value*100, breaks=c(-Inf, levels[3], levels[2], levels[1], Inf), label=c("***", "**", "*", ""))
plot.data$variable <- revalue(plot.data$variable, replace=c('Class.1' = 'Class 1',
                                                            'Class.2' = 'Class 2',
                                                            'Class.3' = 'Class 3',
                                                            'Class.4' = 'Class 4',
                                                            'Class.5' = 'Class 5'))

# figure
fig_tmp <- ggplot(aes(x=variable, y=Factor, fill=-log10(value)), data=plot.data)
fig_AB <- fig_tmp + geom_tile() + 
  scale_fill_gradient(low="white", high="royalblue3") + 
  geom_text(aes(label=stars), color="black", size=5) + 
  labs(y=NULL, x=NULL, fill=expression(-log[10]('p-val'))) +
  scale_y_discrete(limits=rev) +
  theme_bw() + theme(axis.text.x=element_text(angle = 45, hjust = 1)) +
  facet_wrap(~Type) 
  
fig_AB
