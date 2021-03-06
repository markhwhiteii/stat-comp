library(tidyverse)
library(ggridges)
dat <- read_csv("../data/results_df.csv") %>% 
  bind_rows(read_csv("../data/results_df_101-150.csv")) %>% 
  bind_rows(read_csv("../data/results_df_151-200.csv")) %>% 
  bind_rows(read_csv("../data/results_df_201-300.csv")) %>% 
  bind_rows(read_csv("../data/results_df_301-350.csv")) %>% 
  bind_rows(read_csv("../data/results_df_351-450.csv")) %>% 
  bind_rows(read_csv("../data/results_df_451-500.csv")) %>% 
  separate(v1, c("algorithm", "sampling"), "_", FALSE) %>% 
  mutate(
    v1 = as.factor(v1),
    algorithm = factor(
      algorithm, levels = c("c50", "adaboost", "randomforest", "xgboost")
      ),
    sampling = as.factor(sampling),
    predict_vars = linear_vars + 5,
    iter = as.factor(iter),
    spec = tn / (tn + fp),
    auc_roc = (1 + rec - (1 - spec)) / 2
  )
names(dat)[1] <- "model"
dat <- dat[, !names(dat) %in% c("cor_vars", "linear_vars")]

glimpse(dat)
summary(dat)

## proportion zeros
prop_zeros <- dat %>% 
  mutate(total_pos = tp + fp) %>% 
  group_by(model) %>% 
  summarise_at("total_pos", funs(mean(. == 0))) %>% 
  arrange(desc(total_pos))

prop_f1_nan <- dat %>% 
  mutate(f1_nan = is.nan(f1)) %>% 
  group_by(model) %>% 
  summarise_at("f1_nan", funs(mean(. == TRUE))) %>% 
  arrange(desc(f1_nan))

good_models <- prop_f1_nan %>% 
  filter(f1_nan == 0) %>% 
  pull(model) %>% 
  as.character()

## summarize results
mean_results <- dat %>% 
  filter(model %in% good_models) %>% 
  select(model, prec, rec, f1, auc_roc) %>% 
  group_by(model) %>% 
  summarise_if(is.numeric, mean)

mean_results_all <- dat %>% 
  select(model, prec, rec, spec, f1, auc_roc) %>% 
  group_by(model) %>% 
  summarise_if(is.numeric, mean, na.rm = TRUE)

## correlation between precision and recall
prec_rec_cors <- sapply(good_models, function(x) {
  temp <- dat[dat$model == x, ]
  cor(temp$prec, temp$rec)
})

tibble(good_models, prec_rec_cors) %>% 
  arrange(desc(prec_rec_cors))

ggplot(dat[dat$model %in% good_models, ], 
       aes(x = prec, y = rec, colour = model)) +
  geom_jitter() +
  geom_smooth(method = "lm", se = FALSE)
 
## bivariate correlations
dat[unique(dat$iter) ,c("prec", "rec", "f1", "n", "noise_vars", 
       "predict_vars", "minority_size")] %>% 
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
  filter(model %in% good_models) %>% 
  group_by(algorithm, sampling) %>% 
  summarise_at(vars(prec, rec, f1, auc_roc), 
               funs(mean(., na.rm = TRUE)))

plot_precision <- ggplot(dat_summary, 
                         aes(x = algorithm, y = prec, fill = sampling)) +
  geom_bar(stat = "identity", position = "dodge") +
  theme_light()

plot_recall <- ggplot(dat_summary, 
                      aes(x = algorithm, y = rec, fill = sampling)) +
  geom_bar(stat = "identity", position = "dodge") +
  theme_light()

plot_f1 <- ggplot(dat_summary, aes(x = algorithm, y = f1, fill = sampling)) +
  geom_bar(stat = "identity", position = "dodge") +
  theme_light()

# change to ridge plot, because incorporates variance
good_dat <- dat[dat$model %in% good_models, ]

# precision
ranked_names <- names(sort(tapply(
  good_dat$prec, droplevels(good_dat$model), mean, na.rm = TRUE),
  decreasing = TRUE
))
good_dat$model <- factor(good_dat$model, levels = c(ranked_names))
ridge_prec <- ggplot(good_dat, aes(y = model, x = prec)) +
  geom_density_ridges(alpha = .7) +
  theme_light() +
  labs(x = "Precision", y = "Model")

# recall
ranked_names <- names(sort(tapply(
  good_dat$rec, droplevels(good_dat$model), mean, na.rm = TRUE), ## WHY SOME NA?
  decreasing = TRUE
))
good_dat$model <- factor(good_dat$model, levels = c(ranked_names))
ridge_rec <- ggplot(good_dat, aes(y = model, x = rec)) +
  geom_density_ridges(alpha = .7) +
  theme_light() +
  labs(x = "Recall", y = "Model")

# f1
ranked_names <- names(sort(tapply(
  good_dat$f1, droplevels(good_dat$model), mean, na.rm = TRUE),
  decreasing = TRUE
))
good_dat$model <- factor(good_dat$model, levels = c(ranked_names))
ridge_f1 <- ggplot(good_dat, aes(y = model, x = f1)) +
  geom_density_ridges(alpha = .7) +
  theme_light() +
  labs(x = "F1 Score", y = "Model")

## all in one plot
dat_long_ridge <- good_dat %>% 
  transmute(model, auc_roc, f1, prec, rec) %>% 
  gather(metric, value, -model)
ridge_all <- ggplot(dat_long_ridge, aes(y = model, x = value)) +
  geom_density_ridges(alpha = .7) +
  facet_wrap( ~ metric, scales = "free_x") +
  theme_light() +
  labs(x = NULL, y = "Model") +
  theme(text = element_text(size = 14))

## trends with each model
plot_n_f1 <- ggplot(dat[dat$model %in% good_models, ], 
                    aes(x = n, y = f1, colour = model)) +
  geom_jitter() + 
  geom_smooth(method = "lm", se = FALSE)

plot_minoritysize_f1 <- ggplot(dat[dat$model %in% good_models, ], 
                    aes(x = minority_size, y = f1, colour = model)) +
  geom_jitter() + 
  geom_smooth(method = "lm", se = FALSE)

## also want to look at the variance of each score
rec_sd <- with(dat[dat$model %in% good_models, ], tapply(rec, model, sd, na.rm = TRUE))
prec_sd <- with(dat[dat$model %in% good_models, ], tapply(prec, model, sd, na.rm = TRUE))
f1_sd <- with(dat[dat$model %in% good_models, ], tapply(f1, model, sd, na.rm = TRUE))

score_sds <- data.frame(rec_sd, prec_sd, f1_sd) %>% 
  lapply(round, 3) %>% 
  as.data.frame() %>% 
  rownames_to_column("model") %>% 
  filter(model %in% good_models)

## correlation between characteristics of the data and scores for each model
vars <- c("n", "minority_size", "predict_vars", "noise_vars")
auc_roc_cors <- lapply(good_models, function(x) {
  temp <- dat[(dat$model == x) & 
                (dat$model %in% good_models) & 
                (!is.nan(dat$f1)), ]
  cor(temp[, c("auc_roc", vars)])[-1, 1]
})
names(auc_roc_cors) <- good_models

f1_cors <- lapply(good_models, function(x) {
  temp <- dat[(dat$model == x) & 
                (dat$model %in% good_models) & 
                (!is.nan(dat$f1)), ]
  cor(temp[, c("f1", vars)])[-1, 1]
})
names(f1_cors) <- good_models

prec_cors <- lapply(good_models, function(x) {
  temp <- dat[(dat$model == x) & 
                (dat$model %in% good_models) & 
                (!is.nan(dat$prec)), ]
  cor(temp[, c("prec", vars)])[-1, 1]
})
names(prec_cors) <- good_models

rec_cors <- lapply(good_models, function(x) {
  temp <- dat[(dat$model == x) & 
                (dat$model %in% good_models) & 
                (!is.nan(dat$rec)), ]
  cor(temp[, c("rec", vars)])[-1, 1]
})
names(rec_cors) <- good_models

auc_cor_dat <- dat[dat$model %in% good_models, ] %>% 
  

auc_n
auc_minority_size
auc_predictors
auc_noise

