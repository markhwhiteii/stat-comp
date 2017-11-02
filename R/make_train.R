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
