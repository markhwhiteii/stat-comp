library(tidyverse)
dat <- read_csv("../data/results_df.csv") %>% 
  mutate(
    iter_number = as.factor(iter_number),
    algorithm   = factor(
      algorithm, levels = c("c50", "adaboost", "randomforest", "xgboost")
    ),
    sampling    = as.factor(sampling)
  )

glimpse(dat)
summary(dat)

dat[,c("precision", "recall", "f1", "n", "noise_vars", 
       "corr_vars", "linear_vars", "minority_size")] %>% 
  cor(use = "pairwise.complete.obs") %>% 
  round(3)

## compare by algorithm
model_algorithm <- lm(f1 ~ algorithm, dat)
summary(model_algorithm)

## compare by sampling technique
model_sampling <- lm(f1 ~ sampling, dat)
summary(model_sampling)

## both together
dat_summary <- dat %>% 
  group_by(algorithm, sampling) %>% 
  summarise_at(vars(precision, recall, f1), funs(mean(., na.rm = TRUE)))

plot_precision <- ggplot(dat_summary, aes(x = algorithm, y = precision, fill = sampling)) +
  geom_bar(stat = "identity", position = "dodge") +
  theme_light()

plot_recall <- ggplot(dat_summary, aes(x = algorithm, y = recall, fill = sampling)) +
  geom_bar(stat = "identity", position = "dodge") +
  theme_light()

plot_f1 <- ggplot(dat_summary, aes(x = algorithm, y = f1, fill = sampling)) +
  geom_bar(stat = "identity", position = "dodge") +
  theme_light()
