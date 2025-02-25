library(ggplot2)

df <- expand.grid(Views = 1:3, Factors=1:5)
df$Active <- factor(c(1,1,1,
               0,1,1,
               1,0,1,
               1,1,0,
               0,1,0))
df$Views <- factor(df$Views)
df$Factors <- factor(df$Factors)


fig <- ggplot(df, 
       aes(x = Factors, y = Views, fill = Active)) + 
  geom_tile(color = "white",
            lwd = 1.5,
            linetype = 1) + 
  scale_fill_manual(values=c('black', 'lightgray'),name = "Factor-view activity") +
  theme_bw()
fig
