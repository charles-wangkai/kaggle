# Plots the means and standard deviations of pixels by digit in the training set

library(ggplot2)
library(grid)
library(readr)

train <- data.frame(read_csv("../data/train.csv"))

labels   <- train[,1]
features <- train[,-1]

means <- aggregate(features, list(labels), mean)

stds <- aggregate(features, list(labels), sd)

rowsToPlot <- 1:10

rowToMatrix <- function(row) {
  intensity <- as.numeric(row)/max(as.numeric(row))
  return(t(matrix((rgb(intensity, intensity, intensity)), 28, 28)))
}

geom_digit <- function(digits) {
  layer(geom = GeomRasterDigit, stat = "identity", position = "identity", data = NULL,
        params = list(digits=digits))
}

GeomRasterDigit <- ggproto("GeomRasterDigit",
                           ggplot2:::GeomRaster,
                           draw_panel = function(data, panel_scales, coordinates, digits = digits) {
                             corners <- data.frame(x = c(-Inf, Inf), y = c(-Inf, Inf))
                             bounds <- coordinates$transform(corners, panel_scales)
                             x_rng <- range(bounds$x, na.rm = TRUE)
                             y_rng <- range(bounds$y, na.rm = TRUE)
                             rasterGrob(as.raster(rowToMatrix(digits[data$rows,])),
                                        x_rng[1], y_rng[1], diff(x_rng), diff(y_rng), 
                                        default.units = "native", just = c("left","bottom"), 
                                        interpolate = FALSE)
                           })

blank <- theme(strip.background = element_blank(),
               strip.text.x = element_blank(),
               axis.text.x = element_blank(),
               axis.text.y = element_blank(),
               axis.ticks = element_blank(),
               axis.line = element_blank())

p <- ggplot(data.frame(rows=rowsToPlot, labels=means[,1]), aes(x=.1, y=.9, rows=rows, label=labels)) + geom_blank() + xlim(0,1) + ylim(0,1) + xlab("") + ylab("") + 
  facet_wrap(~ rows, ncol=5) +
  geom_digit(means[,-1]) +
  geom_text(colour="#53cfff") +
  blank +
  ggtitle("Pixel Means")

ggsave("pixel_means.png", p)

p <- ggplot(data.frame(rows=rowsToPlot, labels=stds[,1]), aes(x=.1, y=.9, rows=rows, label=labels)) + geom_blank() + xlim(0,1) + ylim(0,1) + xlab("") + ylab("") + 
  facet_wrap(~ rows, ncol=5) +
  geom_digit(stds[,-1]) +
  geom_text(colour="#53cfff") +
  blank +
  ggtitle("Pixel Standard Deviations")

ggsave("pixel_stds.png", p)