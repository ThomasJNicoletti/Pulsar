# The purpose of this script is to provide insight on how multivariate outliers were defined prior to beginning the Pulsar project.

# First, Mahalanobis distances were calculated for the predictor variables.
# Second, the original Pulsar and new Mahalanobis datasets were joined together into a dataframe.
# Third, based on the chi-square critical value for eight variables, multivariate outliers were detected and removed.
# Finally, Excel formulas were used to categorize candidates from the original Pulsar dataset into a new binary outcome variable.

install.packages('stats')

Pulsar <- read.csv('Pulsar.csv', header = TRUE)

Mahalanobis <- mahalanobis(Pulsar[, 1:8], colMeans(Pulsar[, 1:8]), cov(Pulsar[, 1:8]))
Pulsar_Mahalanobis <- data.frame(Pulsar[, 1:8], Mahalanobis)
Pulsar_No_Outliers <- Pulsar_Mahalanobis[which(Pulsar_Mahalanobis$Mahalanobis < 26.13), ]
nrow(Pulsar_No_Outliers)

write.csv(Pulsar_No_Outliers, 'Pulsar_Revised.csv')
