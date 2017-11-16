library(tidyverse)
library(ggridges)
dat <- read_csv("../data/results_df.csv") %>% 
  bind_rows(read_csv("../data/results_df_101-150.csv")) %>% 
  bind_rows(read_csv("../data/results_df_151-200.csv")) %>% 
  bind_rows(read_csv("../data/results_df_201-300.csv")) %>% 
  bind_rows(read_csv("../data/results_df_301-350.csv")) %>% 
  bind_rows(read_csv("../data/results_df_351-450.csv")) %>% 
  separate(v1, c("algorithm", "sampling"), "_", FALSE) %>% 
  mutate(
    v1 = as.factor(v1),
    algorithm = factor(
      algorithm, levels = c("c50", "adaboost", "randomforest", "xgboost")
      ),
    sampling = as.factor(sampling),
    iter = as.factor(iter)
  )
names(dat)[1] <- "model"

glimpse(dat)
summary(dat)

## proportion zeros
prop_zeros <- dat %>% 
  mutate(total_pos = fp + fp) %>% 
  group_by(model) %>% 
  summarise_at("total_pos", funs(mean(. == 0))) %>% 
  arrange(desc(total_pos))

good_models <- prop_zeros %>% 
  filter(total_pos == 0) %>% 
  pull(model) %>% 
  as.character()

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
       "cor_vars", "linear_vars", "minority_size")] %>% 
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
  summarise_at(vars(prec, rec, f1), funs(mean(., na.rm = TRUE)))

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
  good_dat$prec, droplevels(good_dat$model), mean, na.rm = TRUE), ## WHY SOME NA?
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
  transmute(model, f1, prec, rec) %>% 
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
