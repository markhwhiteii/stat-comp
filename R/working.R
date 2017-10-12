library(caret)
library(unbalanced)
library(randomForest)

## function to fix between zero and one
normalize <- function(x) {
  ((x - min(x)) / (max(x) - min(x)))
}

## set params
n <- 100000
pos_class <- .025

## data from numerous distributions
data <- matrix(data = c(
  rbeta(n, shape1 = runif(1, .1, 3), shape2 = runif(1, .1, 3)),
  rcauchy(n),
  rchisq(n, df = runif(1, .1, 3)),
  rexp(n),
  rf(n, df1 = runif(1, .1, 3), df2 = runif(1, .1, 3)),
  rgamma(n, shape = runif(1, .1, 3)),
  rlnorm(n),
  rt(n, df = runif(1, .1, 3)),
  rnorm(n)
), nrow = n)

## get predicted
outcome <- rowSums(data)
qtiles <- unname(quantile(outcome, pos_class))
outcome <- factor(ifelse(outcome < qtiles, 1, 0))

## sampling
data_smote <- ubSMOTE(data, outcome)

## model
rf_smote <- randomForest(data_smote$X, factor(data_smote$Y))





