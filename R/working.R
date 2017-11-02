ps <- c("caret", "unbalanced", "randomForest", "fastAdaboost", 
        "xgboost", "C50", "dplyr", "tidyr", "stringr", "readr")
fs <- c("two_class_sim.R", "get_results.R", "make_train.R", "make_test.R")
invisible(lapply(ps, library, character.only = TRUE))
invisible(lapply(fs, source))

## set parameters
set.seed(1839)
n_iter       <- 1000
pred         <- round(rnorm(n_iter, 50, 7))
prop_noise   <- abs(rnorm(n_iter, .15, .033))
tot_noise    <- round(pred * prop_noise)
n            <- round(rnorm(n_iter, 40000, 5000))
noiseVars    <- round(tot_noise / 2)
corrVars     <- tot_noise - noiseVars
linearVars   <- pred - tot_noise
minoritySize <- rnorm(n_iter, .03, .007)
seeds        <- as.integer(runif(n_iter, 1, 100000000))

results <- vector("list", 1600)
counter <- 0
for (iter in 1:100) {
  # generate data
  set.seed(seeds[iter])
  dat <- two_class_sim(
    n            = n[iter],
    intercept    = 0,
    linearVars   = linearVars[iter],
    noiseVars    = noiseVars[iter],
    corrVars     = corrVars[iter],
    minoritySize = minoritySize[iter]
  )
  cases         <- sample(nrow(dat), nrow(dat) * .7)
  training      <- make_train(dat, cases)
  testing       <- make_test(dat, cases)
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
    counter <- counter + 1
    results[[counter]] <- as.list(c(
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
    counter <- counter + 1
    results[[counter]] <- as.list(c(
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
    counter <- counter + 1
    results[[counter]] <- as.list(c(
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
    counter <- counter + 1
    results[[counter]] <- as.list(c(
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
