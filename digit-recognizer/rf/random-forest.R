library(randomForest)
library(readr)

set.seed(0)

numTrees <- 100

train <- read_csv("../data/train.csv")
test <- read_csv("../data/test.csv")

labels <- as.factor(unlist(train[,1]))
x <- train[,-1]

rf <- randomForest(x, labels, xtest = test, ntree = numTrees, do.trace = TRUE)
cat(sprintf("OOB error rate: %f%%\n", tail(rf$err.rate[, "OOB"], n = 1) * 100))

predictions <- data.frame(ImageId = 1:nrow(test), Label = levels(labels)[rf$test$predicted])

write_csv(predictions, "submission.csv")