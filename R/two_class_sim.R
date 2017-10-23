two_class_sim <- function(n, intercept, linearVars, noiseVars, 
                          corrVars, minoritySize) {
  
  sigma <- matrix(c(2, 1.3, 1.3, 2), 2, 2)
  tmpData <- data.frame(MASS::mvrnorm(n = n, c(0, 0), sigma))
  names(tmpData) <- paste("TwoFactor", 1:2, sep = "")
  
  tmpData <- cbind(tmpData, matrix(rnorm(n * linearVars), ncol = linearVars))
  colnames(tmpData)[(1:linearVars) + 2] <- paste(
    "Linear", gsub(" ", "0", format(1:linearVars)), sep = ""
  )

  tmpData$Nonlinear1 <- runif(n, min = -1)
  tmpData <- cbind(tmpData, matrix(runif(n * 2), ncol = 2))
  colnames(tmpData)[(ncol(tmpData) - 1):ncol(tmpData)] <- paste(
    "Nonlinear", 2:3, sep = ""
  )
  tmpData <- as.data.frame(tmpData)
  p <- ncol(tmpData)
  
  tmpData <- cbind(tmpData, matrix(rnorm(n * noiseVars), ncol = noiseVars))
  colnames(tmpData)[(p + 1):ncol(tmpData)] <- paste(
    "Noise", gsub(" ", "0", format(1:noiseVars)), sep = ""
  )
  
  lp <- intercept - 4 * tmpData$TwoFactor1 + 4 * tmpData$TwoFactor2 + 
    2 * tmpData$TwoFactor1 * tmpData$TwoFactor2 + (tmpData$Nonlinear1^3) + 
    2 * exp(-6 * (tmpData$Nonlinear1 - 0.3)^2) + 
    2 * sin(pi * tmpData$Nonlinear2 * tmpData$Nonlinear3)

  lin <- seq(10, 1, length = linearVars)/4
  lin <- lin * rep(c(-1, 1), floor(linearVars) + 1)[1:linearVars]
  
  for (i in seq(along = lin)) lp <- lp + tmpData[, i + 3] * lin[i]
  
  prob <- binomial()$linkinv(lp)
  prob <- prob + 
    (minoritySize - sort(prob)[(1 - minoritySize) * length(sort(prob))])
  tmpData$Class <- ifelse(prob <= runif(n), "Class1", "Class2")
  tmpData$Class <- factor(tmpData$Class, levels = c("Class1", "Class2"))

  tmpData
}
