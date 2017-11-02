make_test <- function(dat, cases) {
  list(
    X = dat[-cases, -ncol(dat)], 
    y = as.factor(ifelse(dat[-cases, ncol(dat)] == "Class2", 1, 0))
  )
}
