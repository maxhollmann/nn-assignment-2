library(ggplot2)

setwd("/home/max/dev/nn/assignment2/3")
out_dir = "duranium/out"

model_f = paste(out_dir, "models.csv", sep = "/")
if (file.exists(model_f)) {
  models = read.csv(model_f, stringsAsFactors = F)
  models = models[, names(models) != "X"]
} else {
  models = data.frame()
}

for (model in list.files(out_dir, include.dirs = T, full.names = T)) {
  for (version in list.files(model, include.dirs = T, full.names = T)) {
    for (acc_csv in list.files(version, "*_accuracy.csv", full.names = T)) {
      d = read.csv(acc_csv)
      d$epoch = d$epoch + 1
      d$file = sub("accuracy.csv$", "", acc_csv)
      last = d[nrow(d), ]
      
      if (paste(last$model, last$params) %in% paste(models$model, models$params))
        next
      
      models = rbind(models, last)
      
      p = ggplot(d, aes(epoch, acc)) + theme_bw() + 
        labs(x = "Epoch", y = "Test Accuracy") +
        geom_line()
      print(p)
      
      base_f = sub("accuracy\\.csv$", "", acc_csv)
      ggsave(paste0(base_f, "accuracy.png"), p)
    }
  }
}

models = models[order(models$acc), ]

stop()


write.csv(models, model_f, row.names = F)


coa <- function(...) {
  Reduce(function(x, y) {
    i <- which(is.null(x))
    x[i] <- y[i]
    x},
    list(...))
}


d = data.frame()
for (i in 1:nrow(models)) {
  m = models[i, ]
  
  if (startsWith(m$params, "{")) {
    p = fromJSON(m$params)
    d = rbind(d, data.frame(acc = m$acc, nwords = p$n_words, lr = p$lr, bs = p$bs, epochs = p$epochs, 
                            hidden = do.call("paste0", list(p$hidden, sep = "-"))))
  } else {
    match = str_extract_all(m$params, "[a-z]+=[^_$]+")
    l = list()
    #print(match)
    for (match in match[[1]]) {
      vars = strsplit(match, "=")[[1]]
      l[vars[1]] = vars[2]
    }
    print(list(acc = m$acc, nwords = l$words, lr = l$lr, bs = l$bs, epochs = l$epochs, hidden = l$hidden))
    
    d = rbind(d, data.frame(acc = m$acc, nwords = l$words, lr = l$lr, bs = l$bs, epochs = l$epochs, hidden = l$hidden))
  }
  
}

d[, names(d) != "acc"] = data.frame(lapply(d[, names(d) != "acc"], as.factor))

#print(summary(lm(acc ~ nwords + lr + epochs, d)))

