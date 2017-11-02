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
