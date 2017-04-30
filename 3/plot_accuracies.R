library(ggplot2)

setwd("/home/max/dev/nn/assignment2/3")

for (model in list.files("out", include.dirs = T, full.names = T)) {
  for (version in list.files(model, include.dirs = T, full.names = T)) {
    for (acc_csv in list.files(version, "*_accuracy.csv", full.names = T)) {
      d = read.csv(acc_csv)
      
      p = ggplot(d, aes(epoch, acc)) + theme_bw() + 
        labs(x = "Epoch", y = "Test Accuracy") +
        geom_line()
      print(p)
      
      acc_png = sub("\\.csv$", ".png", acc_csv)
      ggsave(acc_png, p)
    }
  }
}

