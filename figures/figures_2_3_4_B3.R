library(ggplot2)
library(dplyr)
library(tidyverse)
library(plyr)
library(ggthemes)
library(ggpubr)
library(patchwork)

# colors
col_pal <- c("#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442",
             "#0072B2", "#D55E00", "#CC79A7", "#B276B2", "#60BD68")

color_fa_ctm <- c('FA(Oracle)'=col_pal[1], 'FACTM'=col_pal[7], 'FA+CTM'=col_pal[2], 'MOFA'=col_pal[5], 
                  'muVI'=col_pal[4], 'PCA'=col_pal[3], 'Tucker'=col_pal[6], 'LDA'=col_pal[9], 'ProdLDA'=col_pal[10], 'CTM'=col_pal[2])

# Figure 2 - fa part ----

df <- read.csv('data//simulations//simulation_results_fa.csv')

df$Models <- factor(df$Models)
df$Models <- revalue(df$Models, c('FA'='del_fa', 'FA(Oracle)'='FA(Oracle)', 'FA+CTM'='FA+CTM', "FACTM"='del_factm','FACTM(R)'='FACTM',
            "MOFA"="MOFA", "muVI"="del_muVI", "muVI_prior"='muVI', "PCA"='PCA',"Tucker"='Tucker'))
df <- df %>% filter(! Models %in% c('del_fa', "del_muVI", 'del_factm') )
df$Models <- factor(df$Models, levels=c('FA(Oracle)', 'FACTM', 'FA+CTM', 'MOFA', 'muVI', 'PCA', 'Tucker'))

df$sim_scenario <- factor(df$sim_scenario)
df$sim_scenario <- revalue(df$sim_scenario, c('scaling_D_topics'='Scenario 3', 'scaling_topics_param'='Scenario 2', 'scaling_weights'='Scenario 1'))
df$sim_scenario <- factor(df$sim_scenario, levels=c('Scenario 1', 'Scenario 2', 'Scenario 3'))
df[df$sim_scenario == 'Scenario 3','param'] <- df[df$sim_scenario == 'Scenario 3','param']*10

df$param <- as.numeric(df$param)
df$param[df$sim_scenario == 'Scenario 1'] <- sapply(df$param[df$sim_scenario == 'Scenario 1'], function(x0) paste('lambda =', x0))
df$param[df$sim_scenario == 'Scenario 2'] <- sapply(df$param[df$sim_scenario == 'Scenario 2'], function(x0) paste('alpha =', ifelse(as.numeric(x0) < 10, paste0('0', x0), paste0(x0))))
df$param[df$sim_scenario == 'Scenario 3'] <- sapply(df$param[df$sim_scenario == 'Scenario 3'], function(x0) paste('L =', ifelse(as.numeric(x0) < 10, paste0('0', x0), paste0("",x0))))

fig_2 <- ggplot(df, aes(x=factor(param), y=value, fill=Models)) +
  geom_boxplot() +
  scale_fill_manual(values=color_fa_ctm)+
  facet_grid(~sim_scenario, scales='free_x') +
  scale_x_discrete(labels=c("lambda = 0" = expression(lambda == 0),
                            "lambda = 0.5" = expression(lambda == 0.5),
                            "lambda = 1" = expression(lambda == 1),
                            "lambda = 1.5" = expression(lambda == 1.5),
                            "lambda = 2" = expression(lambda == 2),
                            "alpha = 01" = expression(alpha == 1),
                            "alpha = 05" = expression(alpha == 5),
                            "alpha = 10" = expression(alpha == 10),
                            "L = 05" = expression(L == 5),
                            "L = 10" = expression(L == 10),
                            "L = 15" = expression(L == 15)))+
  labs(x = NULL) +
  ylab('Spearman correlation') +
  geom_hline(yintercept=1, linetype='dotted')+
  theme_bw() +
  theme(legend.position="bottom", legend.margin=margin(0,0,0,0),
        legend.box.margin=margin(-5,-5,-5,-5)) +
  guides(fill = guide_legend(nrow = 1))
fig_2


# Figure 3 - structered view - part 1 ----

df = read.csv('data//simulations//simulation_results_ctm_v1.csv')

df <- df %>% filter(param!=0)

df$sim_scenario <- factor(df$sim_scenario)
df$sim_scenario <- factor(df$sim_scenario, levels=c('scaling_weights', 'scaling_topics_param', 'scaling_D_topics'))
df$sim_scenario <- revalue(df$sim_scenario, c('scaling_D_topics'=expression(paste('Scenario 3')), 
                                              'scaling_topics_param'=expression(paste('Scenario 2')), 
                                              'scaling_weights'=expression(paste('Scenario 1'))))

df$param <- as.numeric(df$param)
df$param[as.numeric(df$sim_scenari) == 1] <- sapply(df$param[as.numeric(df$sim_scenari) == 1], function(x0) paste('lambda =', x0))
df$param[as.numeric(df$sim_scenari) == 2] <- sapply(df$param[as.numeric(df$sim_scenari) == 2], function(x0) paste('alpha =', ifelse(as.numeric(x0) < 10, paste0('0', x0), paste0(x0))))
df$param[as.numeric(df$sim_scenari) == 3] <- sapply(df$param[as.numeric(df$sim_scenari) == 3], function(x0) paste('L =', ifelse(as.numeric(x0) < 10, paste0('0', x0), paste0("",x0))))

df$Models <- factor(df$Models)
df$Models <- revalue(df$Models, c("FACTM"='del_factm','FACTM(R)'='FACTM',
                                  "CTM"="CTM", "LDA"="LDA", "ProdLDA"='ProdLDA'))
df <- df %>% filter(! Models %in% c('del_factm') )
df$Models <- factor(df$Models, levels=c('FACTM', 'CTM', 'LDA', 'ProdLDA'))

df <- df %>% filter(var  %in% c('muFA_corr_spearmann', 'topics_corr_spearmann', 'clusters_ARI'))

df$var <- factor(df$var)
df$var <- revalue(df$var, c("muFA_corr_spearmann"=expression(paste(mu[FA], ' (Spearman corr)')),
                            'topics_corr_spearmann'=expression(paste(beta, ' (Spearman corr)')),
                             "clusters_ARI"=expression(paste(xi, ' (ARI)'))))

df$yintercept_plot <- NA
df$yintercept_plot[as.numeric(df$var) == 1 | as.numeric(df$var) == 3] = 1

fig_3 <- ggplot(df, aes(x=factor(param), y=value, fill=Models)) +
  geom_boxplot() +
  scale_fill_manual(values=color_fa_ctm)+
  scale_x_discrete(labels=c("lambda = 0" = expression(lambda == 0),
                            "lambda = 0.5" = expression(lambda == 0.5),
                            "lambda = 1" = expression(lambda == 1),
                            "lambda = 1.5" = expression(lambda == 1.5),
                            "lambda = 2" = expression(lambda == 2),
                            "alpha = 01" = expression(alpha == 1),
                            "alpha = 05" = expression(alpha == 5),
                            "alpha = 10" = expression(alpha == 10),
                            "L = 05" = expression(L == 5),
                            "L = 10" = expression(L == 10),
                            "L = 15" = expression(L == 15)))+
  facet_grid(var~sim_scenario, scales='free', labeller = label_parsed) +
  xlab(expression(paste("Parameters: ", lambda, ', ', alpha, ', and ', L))) +
  ylab(' ') +
  labs(x = NULL) +
  geom_hline(aes(yintercept=yintercept_plot), linetype='dotted')+
  theme_bw() +
  theme(legend.position="bottom", legend.margin=margin(0,0,0,0),
        legend.box.margin=margin(-5,-5,-5,-5)) 
fig_3

# Figure 4 - structered view - part 2 (population-level variables) ----

df <- read.csv('data//simulations//simulation_results_ctm_v2.csv')

df$Models <- factor(df$Models)
df$Models <- revalue(df$Models, c("FACTM"='del_factm','FACTM(R)'='FACTM',
                                  "CTM"="CTM", "LDA"="LDA", "ProdLDA"='ProdLDA'))
df$Models <- factor(df$Models, levels=c('FACTM', 'CTM'))

df$sim_scenario <- factor(df$sim_scenario)
df$sim_scenario <- factor(df$sim_scenario, levels=c('scaling_mu0', 'scaling_Sigma0'))
df$sim_scenario <- revalue(df$sim_scenario, c('scaling_mu0'=expression(paste('Scenario 4')), 'scaling_Sigma0'=expression(paste('Scenario 5'))))

df$var <- factor(df$var)
df$var <- factor(df$var, levels=c('mu0_corr', 'Sigma0_corr_Frobenius', 'Sigma0_Frobenius'))
df$var <- revalue(df$var, c("mu0_corr"=expression(paste(mu^(0), " (Spearman corr.)")),
                            'Sigma0_corr_Frobenius'=expression(paste(tilde(Sigma)^(0), " (Frobenius norm rel.)")),
                            "Sigma0_Frobenius"=expression(paste(Sigma^(0), " (Frobenius norm rel.)"))))

df <- df %>% filter(Models %in% c('FACTM', 'CTM'))

p1 <- ggplot(df %>% filter(as.integer(df$sim_scenario) == 1, as.integer(df$var) == 1, param > 0), aes(x=factor(param), y=value, fill=Models)) +
  geom_boxplot() +
  scale_fill_manual(values=color_fa_ctm)+
  facet_grid(var~sim_scenario, scales='free_x', labeller = label_parsed) +
  xlab(bquote(lambda[mu^(0)])) +
  labs(y = NULL) +
  geom_hline(yintercept=1, linetype='dotted')+
  theme_bw()  + theme(legend.position="none")

p2 <- ggplot(df %>% filter(as.integer(df$sim_scenario) == 1, as.integer(df$var) == 2, param > 0), aes(x=factor(param), y=value, fill=Models)) +
  geom_boxplot() +
  scale_fill_manual(values=color_fa_ctm)+
  facet_grid(var~sim_scenario, scales='free_x', labeller = label_parsed) +
  xlab(expression(paste(lambda[mu^(0)]))) +
  labs(y = NULL) +
  geom_hline(yintercept=0, linetype='dotted')+
  theme_bw() + theme(legend.position="none")

p3 <- ggplot(df %>% filter(as.integer(df$sim_scenario) == 2, as.integer(df$var) == 3, param > 0), aes(x=factor(param), y=value, fill=Models)) +
  geom_boxplot() +
  scale_fill_manual(values=color_fa_ctm)+
  facet_grid(var~sim_scenario, scales='free_x', labeller = label_parsed) +
  xlab(expression(paste(lambda[Sigma^0]))) +
  labs(y = NULL) +
  geom_hline(yintercept=0, linetype='dotted')+
  theme_bw() + theme(legend.position="none")
p3

p4 <- ggplot(df %>% filter(as.integer(df$sim_scenario) == 2, as.integer(df$var) == 2, param > 0), aes(x=factor(param), y=value, fill=Models)) +
  geom_boxplot() +
  scale_fill_manual(values=color_fa_ctm)+
  facet_grid(var~sim_scenario, scales='free_x', labeller = label_parsed) +
  xlab(expression(paste(lambda[Sigma^0]))) +
  labs(y = NULL) +
  geom_hline(yintercept=0, linetype='dotted')+
  theme_bw()  + theme(legend.position="none")
p4

combined <- p1 + p2 + p3 + p4 & theme(legend.position="bottom", legend.margin=margin(0,0,0,0),
                                      legend.box.margin=margin(-10,-10,-10,-10))
fig_4 <- combined + plot_layout(nrow = 1, ncol = 4, guides = "collect")
fig_4

# Figure B.3 - scenario 6 - sparsity ----

df <- read.csv('data//simulations//simulations_additional_results.csv')

df$Models <- factor(df$Models)
df$Models <- revalue(df$Models, c('FA'='del_fa', 'FA(Oracle)'='FA(Oracle)', 'FA+CTM'='FA+CTM', "FACTM"='del_factm','FACTM(R)'='FACTM',
                                  "mofa"="MOFA", "muVI"="del_muVI", "muVI_prior"='muVI', "PCA"='PCA',"Tucker"='Tucker'))
df <- df %>% filter(! Models %in% c('del_fa', "del_muVI", 'del_factm') )
df$Models <- factor(df$Models, levels=c('MOFA', 'muVI'))

df$sim_scenario <- factor(df$sim_scenario)
df$sim_scenario <- revalue(df$sim_scenario, c('scaling_sparsity'='Scenario 6'))

new_param = seq(0.1, 0.7, length.out=5)
names(new_param) <- rev(sort(unique(df$param)))
df$param <- revalue(factor(df$param), new_param)
df$param <- factor(df$param, levels=new_param)

fig_B3 <- ggplot(df %>% filter(var == 'z_corr_best_order'), aes(x=factor(param), y=value, fill=Models)) +
  geom_boxplot() +
  scale_fill_manual(values=color_fa_ctm)+
  facet_grid(~sim_scenario, scales='free_x') +
  xlab('Feature-wise sparsity fraction') +
  ylab('Spearman correlation') +
  geom_hline(yintercept=1, linetype='dotted')+
  theme_bw() +
  theme(legend.position="right")# +
fig_B3


