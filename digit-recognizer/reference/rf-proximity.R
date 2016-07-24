# Pulling in Ben's improvements (https://www.kaggle.com/users/993/ben-hamner/digit-recognizer/rf-proximity) via copy/paste since at the moment Scripts doesn't support "merging" explicitly.

# Using RandomForest proximity to visualize digits data set

# This script fits a random forest model, and uses "proximity" to visualize the results.
# Proximity between two examples is based on how often they are in a common leaf node.
# The examples are then embedded in R^2 using multidimensional scaling so as to make 
# the Euclidean distances match the proximities from the RF.

# If you're looking for a nice embedding, there are better approaches (e.g. https://lvdmaaten.github.io/tsne/).
# I'm taking this one here since maybe it will provide some insight into what the RF is doing.

set.seed(0)

library(randomForest)
library(ggplot2)
library(MASS)
library(data.table)

numTrees <- 50
numRowsForModel <- 10000
numRowsForMDS <- 1000
numRowsToDrawAsImages <- 750

train <- data.frame(fread("../data/train.csv", header=TRUE))

# Use only a subset of train to save time
smallTrain = train[sample(1:nrow(train), size = numRowsForModel),]
labels <- as.factor(smallTrain[[1]])
smallTrain = smallTrain[,-1]


# Make my own train/test split
inMyTrain = sample(c(TRUE,FALSE), size = numRowsForModel, replace = TRUE)
myTrain = smallTrain[inMyTrain,]
myTest = smallTrain[!inMyTrain,]
labelsMyTrain = labels[inMyTrain]
labelsMyTest = labels[!inMyTrain]

# Random forest (generates proximities)
rf <- randomForest(myTrain, labelsMyTrain, ntree = numTrees, xtest = myTest, proximity = TRUE)
predictions <- levels(labels)[rf$test$predicted]
predictionIsCorrect = labelsMyTest == predictions

cat(sprintf("Proportion correct in my test set: %f\n", mean(predictionIsCorrect)))

# Do MDS on subset (to save time) of proximities
# Get the portion of the proximity matrix just for my holdout set:
prox = rf$test$proximity[,1:nrow(myTest)]

# Get proximities just for a smaller subset:
proxSmall = prox[1:numRowsForMDS,1:numRowsForMDS]
cat("Beginnging MDS (embedding data in R^2, respecting the RF proximities as much as possible:\n")
embeddingSmall = isoMDS(1 - proxSmall, k = 2)

# embeddedSubsetPredictions <- predictions[1:numRowsForMDS]
# embeddedSubsetLabels <- labelsMyTest[1:numRowsForMDS]
# embeddedSubsetPredictionIsCorrect <- predictionIsCorrect[1:numRowsForMDS]

makeAnnotationRaster <- function(rowNum, size, posDF,  imageDF, correct, digit, colors) {
  row <- as.numeric(imageDF[rowNum,])
  t   <- 1 - row/max(row)*0.8
  rowHSV <- if (correct[rowNum]) {
    #hsv((digit[rowNum]+1)/11, 0.4, 1.0, ifelse(t>0.8, 0.0, 0.8))
    rgb(colors[digit[rowNum],"r"],colors[digit[rowNum],"g"],colors[digit[rowNum],"b"],ifelse(t>0.8, 0.0, 0.7))
  } else {
    hsv(0, 1.0, 1.0, ifelse(t>0.8, 0.0, 1.0))
  }
  rowMatrix <- matrix((rowHSV), 28, 28)
  plotSize = ifelse(correct[rowNum], size, size*1.5)
  pos <- c(posDF[rowNum,] - plotSize/2, posDF[rowNum,] + plotSize/2)
  return(annotation_raster(t(rowMatrix), pos[1], pos[3], pos[2], pos[4]))
}

colors = data.frame(r=c(166, 31, 178, 51, 251, 227, 253, 255, 202, 106),
                    g=c(206, 120, 223, 160, 154, 26, 191, 127, 178, 61),
                    b=c(227, 180, 138, 44, 153, 28, 111, 0, 214, 154)) / 255

rowsForPlottingAsImages = sample(1:numRowsForMDS, numRowsToDrawAsImages)
ARs = Map(function(rows) makeAnnotationRaster(rows, .05, embeddingSmall$points, myTest, (predictions == labelsMyTest), as.numeric(labelsMyTest), colors),  
          rowsForPlottingAsImages)

p <- ggplot(data.frame(embeddingSmall$points), aes(x=X1, y=X2)) + 
  geom_blank() + 
  scale_shape_manual(values = c(17,16)) +
  scale_size_manual(values = c(5,3)) +
  labs(color = "True Label ", size="Correctly Classified ", shape="Correctly Classified ") +
  theme_light(base_size=20) +
  theme(strip.background = element_blank(),
        strip.text.x     = element_blank(),
        axis.text.x      = element_blank(),
        axis.text.y      = element_blank(),
        axis.ticks       = element_blank(),
        axis.line        = element_blank(),
        panel.border     = element_blank(),
        legend.position  = "top") +
  xlab("") + ylab("") + 
  ggtitle("2D MNIST Embedding Using RF Proximity\n(red highlights classification errors)")

png(filename = "DigitsEmbedding.png", width = 960, height = 960)
p <- Reduce("+", ARs, init = p)
plot(p)
invisible(dev.off())