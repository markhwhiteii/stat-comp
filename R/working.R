#### NEED TO REFACTOR ####

purrr::walk(
  c("caret", "unbalanced", "randomForest", 
    "fastAdaboost", "xgboost", "C50", "dplyr", "tidyr"), 
  library, character.only = TRUE
)

## create function for accuracy stats
get_results <- function(predicted, actual) {
  conf_matrix <- as.data.frame(table(predicted, actual))
  prec <- conf_matrix[4, 3] / (conf_matrix[4, 3] + conf_matrix[2, 3])
  rec <- conf_matrix[4, 3] / (conf_matrix[4, 3] + conf_matrix[3, 3])
  spec <- conf_matrix[1, 3] / (conf_matrix[1, 3] + conf_matrix[2, 3])
  f1 <- 2 * ((prec * rec) / (prec + rec))
  auroc <- (rec + spec) / 2
  return(c(prec, rec, f1, auroc))
}

## set parameters
set.seed(1839)
pred <- round(rnorm(1, 40, 10), 0)
prop_noise <- abs(rnorm(1, .17, .033))
tot_noise <- round(pred * prop_noise, 0)

n <- 5000 # for practice
#round(rnorm(1, 3000000, 800000), 0) # for real
noiseVars <- round(tot_noise / 2, 0)
corrVars <- tot_noise - noiseVars
linearVars <- pred - tot_noise
intercept <- round(rnorm(1, -30, 1.5), 0)

## simulate data
dat <- twoClassSim(
  n = n,
  noiseVars = noiseVars,
  corrVars = corrVars,
  linearVars = linearVars,
  intercept = intercept
)

## save minority size
minority_size <- prop.table(table(dat$Class))[[2]]

## split into train and test
cases <- sample(nrow(dat), nrow(dat) * .7)
train_X <- dat[cases, -ncol(dat)]
train_y <- as.factor(ifelse(dat[cases, ncol(dat)] == "Class2", 1, 0))
test_X <- dat[-cases, -ncol(dat)]
test_y <- as.factor(ifelse(dat[-cases, ncol(dat)] == "Class2", 1, 0))
rm(dat, cases)

## sampling techniques
# random undersampling
train_under <- ubUnder(train_X, train_y)

# random oversampling
train_over <- ubOver(train_X, train_y)

# SMOTE
train_smote <- ubSMOTE(train_X, train_y)

## train models
# adaboost
ada_under <- adaboost(Class ~ ., mutate(train_under$X, Class = train_under$Y), 10)
ada_over <- adaboost(Class ~ ., mutate(train_over$X, Class = train_over$Y), 10)
ada_smote <- adaboost(Class ~ ., mutate(train_smote$X, Class = train_smote$Y), 10)
ada_none <- adaboost(Class ~ ., mutate(train_X, Class = train_y), 10)

# xgboost
xgb_under <- xgboost(data.matrix(train_under$X), as.numeric(as.character(train_under$Y)),
                     nrounds = 10, verbose = 0, objective = "binary:logistic")
xgb_over <- xgboost(data.matrix(train_over$X), as.numeric(as.character(train_over$Y)), 
                    nrounds = 10, verbose = 0, objective = "binary:logistic")
xgb_smote <- xgboost(data.matrix(train_smote$X), as.numeric(as.character(train_smote$Y)), 
                     nrounds = 10, verbose = 0, objective = "binary:logistic")
xgb_none <- xgboost(data.matrix(train_X), as.numeric(as.character(train_y)), 
                    nrounds = 10, verbose = 0, objective = "binary:logistic")

# random forest
rf_under <- randomForest(train_under$X, train_under$Y)
rf_over <- randomForest(train_over$X, train_over$Y)
rf_smote <- randomForest(train_smote$X, train_smote$Y)
rf_none <- randomForest(train_X, train_y)

# c5.0
c50_under <- C5.0(train_under$X, train_under$Y)
c50_over <- C5.0(train_over$X, train_over$Y)
c50_smote <- C5.0(train_smote$X, train_smote$Y)
c50_none <- C5.0(train_X, train_y)

## memory break
rm(train_under, train_over, train_smote, train_X, train_y)

## predict test set
# adaboost
ada_under_pred <- predict(ada_under, test_X)
ada_over_pred <- predict(ada_over, test_X)
ada_smote_pred <- predict(ada_smote, test_X)
ada_none_pred <- predict(ada_none, test_X)

# xgboost
xgb_under_pred <- as.numeric(predict(xgb_under, data.matrix(test_X)) > .5)
xgb_over_pred <- as.numeric(predict(xgb_over, data.matrix(test_X)) > .5)
xgb_smote_pred <- as.numeric(predict(xgb_smote, data.matrix(test_X)) > .5)
xgb_none_pred <- as.numeric(predict(xgb_none, data.matrix(test_X)) > .5)

# random forest
rf_under_pred <- unname(predict(rf_under, test_X))
rf_over_pred <- unname(predict(rf_over, test_X))
rf_smote_pred <- unname(predict(rf_smote, test_X))
rf_none_pred <- unname(predict(rf_none, test_X))

# c5.0
c50_under_pred <- predict(c50_under, test_X)
c50_over_pred <- predict(c50_over, test_X)
c50_smote_pred <- predict(c50_smote, test_X)
c50_none_pred <- predict(c50_none, test_X)

## metrics
results <- list(
  # adaboost
  ada_under_results = get_results(ada_under_pred$class, test_y), 
  ada_over_results  = get_results(ada_over_pred$class, test_y), 
  ada_smote_results = get_results(ada_smote_pred$class, test_y), 
  ada_none_results  = get_results(ada_none_pred$class, test_y), 
  
  # xgboost
  xgb_under_results = get_results(xgb_under_pred, test_y), 
  xgb_over_results  = get_results(xgb_over_pred, test_y), 
  xgb_smote_results = get_results(xgb_smote_pred, test_y), 
  xgb_none_results  = get_results(xgb_none_pred, test_y), 
  
  # random forest
  rf_under_results = get_results(rf_under_pred, test_y), 
  rf_over_results  = get_results(rf_over_pred, test_y), 
  rf_smote_results = get_results(rf_smote_pred, test_y), 
  rf_none_results  = get_results(rf_none_pred, test_y), 
  
  # c5.0
  c50_under_results = get_results(c50_under_pred, test_y), 
  c50_over_results  = get_results(c50_over_pred, test_y), 
  c50_smote_results = get_results(c50_smote_pred, test_y), 
  c50_none_results  = get_results(c50_none_pred, test_y)
)


## combining results
results_df <- data.frame(model = NA, precision = NA, recall = NA, f1 = NA, auroc = NA)
for (i in 1:length(results)) {
  results_df[i, ] <- c(names(results)[i], results[[i]])
}

## tidying results
results_df <- results_df %>% 
  separate(model, c("algorithm", "sampling", "temp")) %>% 
  select(-temp) %>% 
  mutate(
    n = n,
    noise_vars = noiseVars,
    corr_vars = corrVars,
    linear_vars = linearVars,
    minority_size = minority_size
  )

results_df
