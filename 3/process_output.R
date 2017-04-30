library(ggplot2)

setwd("/home/max/dev/nn/assignment2/3")
out_dir = "duranium/out"

models = data.frame()

for (model in list.files(out_dir, include.dirs = T, full.names = T)) {
  for (version in list.files(model, include.dirs = T, full.names = T)) {
    for (acc_csv in list.files(version, "*_accuracy.csv", full.names = T)) {
      d = read.csv(acc_csv)
      d$epoch = d$epoch + 1
      
      models = rbind(models, d[nrow(d), ])
      
      p = ggplot(d, aes(epoch, acc)) + theme_bw() + 
        labs(x = "Epoch", y = "Test Accuracy") +
        geom_line()
      print(p)
      
      base_f = sub("accuracy\\.csv$", "", acc_csv)
      ggsave(paste0(base_f, "accuracy.png"), p)
    }
  }
}

write.csv(models, paste(out_dir, "models.csv", sep = "/"))


models = models[order(models$acc), ]
