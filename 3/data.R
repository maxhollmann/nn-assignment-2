setwd("/home/max/dev/nn/assignment2/3")

d = read.csv("data.csv")
d = d[d$moderated_role %in% c("guest", "host", "neither"), ]

print(paste("Total cases:", nrow(d)))
print(paste("Guests:", nrow(d[d$moderated_role == "guest", ]), "/", nrow(d[d$moderated_role == "guest", ]) / nrow(d)))
print(paste("Episodes:", length(unique(paste(d$title, d$description)))))

