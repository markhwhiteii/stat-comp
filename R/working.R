setwd("..") # working directory is the project, not the R directory
purrr::walk(
  c("caret", "unbalanced", "randomForest", "fastAdaboost", 
    "xgboost", "C50", "dplyr", "tidyr", "stringr", "readr"), 
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
  return(list(prec, rec, f1, auroc))
}

set.seed(1839)

for (iter in 1:100) {
  ## set parameters
  pred <- round(rnorm(1, 40, 10), 0)
  prop_noise <- abs(rnorm(1, .17, .033))
  tot_noise <- round(pred * prop_noise, 0)
  n <- round(rnorm(1, 30000, 9000), 0)
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
  
  ## memory saving
  rm(dat, cases)
  
  ## sampling techniques
  training <- list(
    # random undersampling
    train_under = ubUnder(train_X, train_y)[-3],
    # random oversampling
    train_over = ubOver(train_X, train_y),
    # SMOTE
    train_smote = ubSMOTE(train_X, train_y),
    # none
    train_none = list(X = train_X, Y = train_y)
  )
  
  ## memory saving
  rm(train_X, train_y)
  
  ## train and test models
  results_df <- data.frame(
    v1 = NA, precision = NA, recall = NA, f1 = NA, auroc = NA
  )
  
  # adaboost
  for (i in seq_along(training)) {
    # set name
    name <- paste0(
      "adaboost_", 
      str_split(names(training[i]), "_", simplify = TRUE)[1,2]
    )
    # create model
    assign(name, adaboost(
      Class ~ ., mutate(training[[i]]$X, Class = training[[i]]$Y), 10
    )
    )
    # predict
    assign(name, predict(get(name), test_X))
    # get results
    results_df <- rbind(results_df, c(name, get_results(get(name)$class, test_y)))
  }
  
  # xgboost
  for (i in seq_along(training)) {
    # set name
    name <- paste0(
      "xgboost_", 
      str_split(names(training[i]), "_", simplify = TRUE)[1,2]
    )
    # create model
    assign(name, xgboost(
      data.matrix(training[[i]]$X), as.numeric(as.character(training[[i]]$Y)),
      nrounds = 10, verbose = 0, objective = "binary:logistic"
    )
    )
    # predict
    assign(name, as.numeric(predict(get(name), data.matrix(test_X))) > .5)
    # get results
    results_df <- rbind(results_df, c(name, get_results(get(name), test_y)))
  }
  
  # random forest
  for (i in seq_along(training)) {
    # set name
    name <- paste0(
      "randomforest_", 
      str_split(names(training[i]), "_", simplify = TRUE)[1,2]
    )
    # create model
    assign(name, randomForest(training[[i]]$X, training[[i]]$Y, ntree = 100))
    # predict
    assign(name, unname(predict(get(name), test_X)))
    # get results
    results_df <- rbind(results_df, c(name, get_results(get(name), test_y)))
  }
  
  # c5.0
  for (i in seq_along(training)) {
    # set name
    name <- paste0(
      "c50_",
      str_split(names(training[i]), "_", simplify = TRUE)[1,2]
    )
    # create model
    assign(name, C5.0(training[[i]]$X, training[[i]]$Y))
    # predict
    assign(name, predict(get(name), test_X))
    # get results
    results_df <- rbind(results_df, c(name, get_results(get(name), test_y)))
  }
  
  rm(test_y)
  
  ## tidying results, saving externally to not keep in working memory
  if (iter == 1) {
    results_df %>% 
      filter(!is.na(v1)) %>% 
      separate(v1, c("algorithm", "sampling"), sep = "_") %>% 
      mutate(
        n = n,
        noise_vars = noiseVars,
        corr_vars = corrVars,
        linear_vars = linearVars,
        minority_size = minority_size,
        iter_number = iter
      ) %>% 
      # first iteration creates the file
      write_csv("data/results_df.csv")
  } else {
    results_df %>% 
      filter(!is.na(v1)) %>% 
      separate(v1, c("algorithm", "sampling"), sep = "_") %>% 
      mutate(
        n = n,
        noise_vars = noiseVars,
        corr_vars = corrVars,
        linear_vars = linearVars,
        minority_size = minority_size,
        iter_number = iter
      ) %>% 
      # all other iterations append to it
      write_csv("data/results_df.csv", append = TRUE)
  }
  print(i)
}
