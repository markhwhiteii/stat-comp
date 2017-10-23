testing <- data.frame(n = NA, linearVars = NA, noiseVars = NA, corrVars = NA, 
                      minoritySize_in = NA, minoritySize_out = NA)

set.seed(1839)

for (i in 1:1000) {
  pred <- round(rnorm(1, 50, 7))
  prop_noise <- abs(rnorm(1, .15, .033))
  tot_noise <- round(pred * prop_noise)
  n <- round(rnorm(1, 40000, 3333))
  noiseVars <- round(tot_noise / 2)
  corrVars <- tot_noise - noiseVars
  linearVars <- pred - tot_noise
  minoritySize <- rnorm(1, .02, .003)
  
  dat <- two_class_sim(
    n = n,
    intercept = 0,
    linearVars = linearVars,
    noiseVars = noiseVars,
    corrVars = corrVars,
    minoritySize = minoritySize
  )
  
  testing[i, ] <- c(n, linearVars, noiseVars, corrVars, 
                    minoritySize, mean(dat$Class == "Class2"))
}

round(cor(testing), 4)

ggplot(testing, aes(x = minoritySize_in, y = minoritySize_out)) +
  geom_point() +
  geom_smooth(method = "loess", se = FALSE)
