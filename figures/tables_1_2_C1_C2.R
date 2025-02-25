library(ggplot2)
library(dplyr)
library(tidyverse)
library(rstatix)
library(xtable)
library(tidyr)
library(plyr)


# Tables 1 and C1 ----

file_path <- 'data//benchmarks//'

df_mirex <- read.csv(paste0(file_path, 'mirex_factors_classification.csv'))
df_mosei <- read.csv(paste0(file_path, 'mosei_factors_classification.csv'))
df_mosi <- read.csv(paste0(file_path, 'mosi_factors_classification.csv'))

df <- rbind(df_mirex, df_mosei)
df <- rbind(df, df_mosi)


df_fa <- df %>% filter(! measure  == 'f1') %>% group_by(measure, dataset, model) %>% 
  dplyr::summarize(mean=mean(value), sd=sd(value)) %>% 
  mutate(tab_val = paste0('$', round(mean, 2), '\\pm ', round(sd, 2), '$')) 
df_fa$model <- factor(df_fa$model)
df_fa$model <- revalue(df_fa$model, c('fa_ctm'='FA+CTM', "factm"='FACTM',
                                  "mofa"="MOFA", "muvi"='muVI'))
df_fa$model <- factor(df_fa$model, levels=c('FACTM', 'FA+CTM', 'MOFA', 'muVI'))

df_fa$measure <- factor(df_fa$measure)
df_fa$measure <- revalue(df_fa$measure, c('roc_auc'='ROC-AUC', "pr_auc"='PR-AUC'))

df_fa <- df_fa %>% select(measure, model, dataset, tab_val) %>% pivot_wider(names_from ='dataset',values_from ='tab_val')
print(xtable(df_fa), include.rownames=F, include.colnames=T, sanitize.text.function=function(x){x})

# Tables 2 and C2 ----

df_mirex <- read.csv(paste0(file_path, 'mirex_structured_classification.csv'))
df_mosei <- read.csv(paste0(file_path, 'mosei_structured_classification.csv'))
df_mosi <- read.csv(paste0(file_path, 'mosi_structured_classification.csv'))


df <- rbind(df_mirex, df_mosei)
df <- rbind(df, df_mosi)

df_ctm <- df %>% filter(! measure  == 'f1') %>% group_by(measure, dataset, model) %>% 
  dplyr::summarize(mean=mean(value), sd=sd(value)) %>% 
  mutate(tab_val = paste0('$', round(mean, 2), '\\pm ', round(sd, 2), '$')) 

df_ctm$model <- factor(df_ctm$model)
df_ctm$model <- revalue(df_ctm$model, c('ctm'='CTM', "factm"='FACTM', 
                                      "lda_log"='LDA'))
df_ctm <- df_ctm %>% filter(model %in% c('CTM', "FACTM", 'LDA'))
df_ctm$model <- factor(df_ctm$model, levels=c("FACTM", 'CTM', 'LDA'))

df_ctm$measure <- factor(df_ctm$measure)
df_ctm$measure <- revalue(df_ctm$measure, c('roc_auc'='ROC-AUC', "pr_auc"='PR-AUC'))

df_ctm <- df_ctm %>% select(measure, model, dataset, tab_val) %>% pivot_wider(names_from ='dataset',values_from ='tab_val')
print(xtable(df_ctm), include.rownames=F, include.colnames=T, sanitize.text.function=function(x){x})

