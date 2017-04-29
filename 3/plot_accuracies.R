library(ggplot2)

setwd("/home/max/dev/nn/assignment2/3")

d = read.csv("out/accuracy.csv")
d$model_inst = factor(paste(d$model, d$params))

for (inst in levels(d$model_inst)) {
  di = d[d$model_inst == inst, ]
  p = ggplot(di, aes(epoch, acc)) + theme_bw() + 
    labs(x = "Epoch", y = "Test Accuracy", title = inst) +
    geom_line()
  print(p)
}
