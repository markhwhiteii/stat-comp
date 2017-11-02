purrr::walk(
  c("caret", "unbalanced", "randomForest", "fastAdaboost", 
    "xgboost", "C50", "dplyr", "tidyr", "stringr", "readr"), 
  library, character.only = TRUE
)

## prepare functions
source("two_class_sim.R")

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
  return(c(tp, fp, tn, fn, prec, rec, f1))
}

make_train <- function(dat, cases) {
  train_X <- dat[cases, -ncol(dat)]
  train_y <- as.factor(ifelse(dat[cases, ncol(dat)] == "Class2", 1, 0))
  list(
    train_under = ubUnder(train_X, train_y)[-3],
    train_over = ubOver(train_X, train_y),
    train_smote = ubSMOTE(train_X, train_y),
    train_none = list(X = train_X, Y = train_y)
  )
}

make_test <- function(dat, cases) {
  list(
    X = dat[-cases, -ncol(dat)], 
    y = as.factor(ifelse(dat[-cases, ncol(dat)] == "Class2", 1, 0))
  )
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

results <- list()
for (iter in 1:100) {
  # generate data
  set.seed(seeds[iter])
  dat <- two_class_sim(
    n = n[iter],
    intercept = 0,
    linearVars = linearVars[iter],
    noiseVars = noiseVars[iter],
    corrVars = corrVars[iter],
    minoritySize = minoritySize[iter]
  )
  cases <- sample(nrow(dat), nrow(dat) * .7)
  training <- make_train(dat, cases)
  testing <- make_test(dat, cases)
  minority_size <- prop.table(table(dat$Class))[[2]]
  
  # adaboost
  for (i in seq_along(training)) {
    name <- paste0(
      "adaboost_", 
      str_split(names(training[i]), "_", simplify = TRUE)[1, 2]
    )
    assign(name, adaboost(
      Class ~ ., mutate(training[[i]]$X, Class = training[[i]]$Y), nIter = 10
    ))
    assign(name, predict(get(name), testing$X))
    results[[length(results) + 1]] <- as.list(c(
      name, get_results(get(name)$class, testing$y), n[iter], noiseVars[iter], 
      corrVars[iter], linearVars[iter], minority_size, iter
    ))
  }
  
  # xgboost
  for (i in seq_along(training)) {
    name <- paste0(
      "xgboost_", 
      str_split(names(training[i]), "_", simplify = TRUE)[1, 2]
    )
    assign(name, xgboost(
      data.matrix(training[[i]]$X), as.numeric(as.character(training[[i]]$Y)),
      nrounds = 10, verbose = 0, objective = "binary:logistic"
    ))
    assign(name, as.numeric(predict(get(name), data.matrix(testing$X)) > .5))
    results[[length(results) + 1]] <- as.list(c(
      name, get_results(get(name), testing$y), n[iter], noiseVars[iter], 
      corrVars[iter], linearVars[iter], minority_size, iter
    ))
  }
  
  # random forest
  for (i in seq_along(training)) {
    name <- paste0(
      "randomforest_", 
      str_split(names(training[i]), "_", simplify = TRUE)[1, 2]
    )
    assign(name, randomForest(training[[i]]$X, training[[i]]$Y, ntree = 100))
    assign(name, unname(predict(get(name), testing$X)))
    results[[length(results) + 1]] <- as.list(c(
      name, get_results(get(name), testing$y), n[iter], noiseVars[iter], 
      corrVars[iter], linearVars[iter], minority_size, iter
    ))
  }
  
  # c5.0
  for (i in seq_along(training)) {
    name <- paste0(
      "c50_",
      str_split(names(training[i]), "_", simplify = TRUE)[1, 2]
    )
    assign(name, C5.0(training[[i]]$X, training[[i]]$Y))
    assign(name, predict(get(name), testing$X))
    results[[length(results) + 1]] <- as.list(c(
      name, get_results(get(name), testing$y), n[iter], noiseVars[iter], 
      corrVars[iter], linearVars[iter], minority_size, iter
    ))
  }
  print(iter)
}

results <- as.data.frame(do.call(rbind, results))
colnames(results) <- c("v1", "tp", "fp", "tn", "fn", "prec", "rec", "f1",
                       "n", "noise_vars", "cor_vars", "linear_vars",
                       "minority_size", "iter")
write_csv(results, "../data/results_df.csv")
