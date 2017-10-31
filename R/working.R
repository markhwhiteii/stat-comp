setwd("..") # working directory is the project, not the R directory
purrr::walk(
  c("caret", "unbalanced", "randomForest", "fastAdaboost", 
    "xgboost", "C50", "dplyr", "tidyr", "stringr", "readr"), 
  library, character.only = TRUE
)

## load in my own simulation function, based on caret
source("R/two_class_sim.R")

## create function for accuracy stats
get_results <- function(predicted, actual) {
  conf_matrix <- as.data.frame(table(predicted, actual))
  tp <- conf_matrix[(conf_matrix[1] == 1) & (conf_matrix[2] == 1), 3] %>% 
    ifelse(length(.) == 0, 0, .)
  fp <- conf_matrix[(conf_matrix[1] == 1) & (conf_matrix[2] == 0), 3] %>% 
    ifelse(length(.) == 0, 0, .)
  tn <- conf_matrix[(conf_matrix[1] == 0) & (conf_matrix[2] == 0), 3] %>% 
    ifelse(length(.) == 0, 0, .)
  fn <- conf_matrix[(conf_matrix[1] == 0) & (conf_matrix[2] == 1), 3] %>% 
    ifelse(length(.) == 0, 0, .)
  prec <- tp / (tp + fp)
  rec <- tp / (tp + fn)
  f1 <- 2 * ((prec * rec) / (prec + rec))
  return(list(tp, fp, tn, fn, prec, rec, f1))
}

## set parameters
set.seed(1839)
n_iter <- 1000
pred <- round(rnorm(n_iter, 50, 7))
prop_noise <- abs(rnorm(n_iter, .15, .033))
tot_noise <- round(pred * prop_noise)
n <- round(rnorm(n_iter, 40000, 5000))
noiseVars <- round(tot_noise / 2)
corrVars <- tot_noise - noiseVars
linearVars <- pred - tot_noise
minoritySize <- rnorm(n_iter, .03, .007)
seeds <- as.integer(runif(n_iter, 1, 100000000))

for (iter in 1:2) {
  ## generating data
  set.seed(seeds[iter])
  dat <- two_class_sim(
    n = n[iter],
    intercept = 0,
    linearVars = linearVars[iter],
    noiseVars = noiseVars[iter],
    corrVars = corrVars[iter],
    minoritySize = minoritySize[iter]
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
    v1 = NA, tp = NA, fp = NA, tn = NA, fn = NA,
    precision = NA, recall = NA, f1 = NA
  )
  
  # adaboost
  for (i in seq_along(training)) {
    # set name
    name <- paste0(
      "adaboost_", 
      str_split(names(training[i]), "_", simplify = TRUE)[1, 2]
    )
    # create model
    assign(name, adaboost(
      Class ~ ., mutate(training[[i]]$X, Class = training[[i]]$Y), nIter = 10
    ))
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
      str_split(names(training[i]), "_", simplify = TRUE)[1, 2]
    )
    # create model
    assign(name, xgboost(
      data.matrix(training[[i]]$X), as.numeric(as.character(training[[i]]$Y)),
      nrounds = 10, verbose = 0, objective = "binary:logistic"
    ))
    # predict
    assign(name, as.numeric(predict(get(name), data.matrix(test_X)) > .5))
    # get results
    results_df <- rbind(results_df, c(name, get_results(get(name), test_y)))
  }
  
  # random forest
  for (i in seq_along(training)) {
    # set name
    name <- paste0(
      "randomforest_", 
      str_split(names(training[i]), "_", simplify = TRUE)[1, 2]
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
      str_split(names(training[i]), "_", simplify = TRUE)[1, 2]
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
        n = n[iter],
        noise_vars = noiseVars[iter],
        corr_vars = corrVars[iter],
        linear_vars = linearVars[iter],
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
        n = n[iter],
        noise_vars = noiseVars[iter],
        corr_vars = corrVars[iter],
        linear_vars = linearVars[iter],
        minority_size = minority_size,
        iter_number = iter
      ) %>% 
      # all other iterations append to it
      write_csv("data/results_df.csv", append = TRUE)
  }
  
  print(iter)
}
