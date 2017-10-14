library(tidyverse)

setwd("..") # set one level up
dat <- read_csv("data/results_df.csv") %>% 
  mutate(
    iter_number = as.factor(iter_number),
    algorithm   = factor(
      algorithm, levels = c("c50", "adaboost", "randomforest", "xgboost")
    ),
    sampling    = as.factor(sampling)
  )

glimpse(dat)

## data characteristics not orthogonal
dat[,c("precision", "recall", "f1", "auroc", "n", "noise_vars", 
       "corr_vars", "linear_vars", "minority_size")] %>% 
  cor(use = "pairwise.complete.obs")

## compare by algorithm
model_algorithm1 <- lm(auroc ~ algorithm, dat)
summary(model_algorithm1) # xgboost does best
model_algorithm2 <- lm(f1 ~ algorithm, dat)
summary(model_algorithm2) # ada and xg do best

## compare by sampling technique
model_sampling1 <- lm(auroc ~ sampling, dat)
summary(model_sampling1) # smote and under do best
model_sampling2 <- lm(f1 ~ sampling, dat)
summary(model_sampling2) # over and none do best?

## both together
dat_summary <- dat %>% 
  group_by(algorithm, sampling) %>% 
  summarise_at(vars(precision, recall, f1, auroc), funs(mean(., na.rm = TRUE)))

plot_precision <- ggplot(dat_summary, aes(x = algorithm, y = precision, fill = sampling)) +
  geom_bar(stat = "identity", position = "dodge") +
  theme_light()

plot_recall <- ggplot(dat_summary, aes(x = algorithm, y = recall, fill = sampling)) +
  geom_bar(stat = "identity", position = "dodge") +
  theme_light()

plot_auroc <- ggplot(dat_summary, aes(x = algorithm, y = auroc, fill = sampling)) +
  geom_bar(stat = "identity", position = "dodge") +
  theme_light()
