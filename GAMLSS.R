df <- read.csv("C:/Users/tlatp/OneDrive/Desktop/리스크 관리 경진 대회/데이터/gamlss_analysis_data.csv")

library(gamlss)
library(gamlss.dist)

# expected_loss(로그변환)
model <- gamlss(
  log_expected_loss ~ .,
  sigma.fo = ~ 1,
  nu.fo = ~ 1,
  family = NO,
  data = df[, !(names(df) %in% c("frequency", "year", "month"))]
)

summary(model)

windows(width = 7, height = 7)
qqnorm(residuals(model))
qqline(residuals(model))

df$fitted_log <- fitted(model)

windows(width = 7, height = 7)
plot(df$fitted_log, df$log_expected_loss,
     xlab = "Predicted (log scale)", ylab = "Actual (log scale)",
     main = "Predicted vs Actual log(expected_loss)")
abline(0, 1, col = "red")

alpha <- 0.99
mu_hat <- fitted(model, "mu")
sigma_hat <- fitted(model, "sigma")
expected_loss_VaR <- exp(mu_hat + qnorm(alpha) * sigma_hat)
summary(expected_loss_VaR)
print(mu_hat)
print(sigma_hat)

windows(width = 7, height = 7)
hist(expected_loss_VaR)


# frequency 
model <- gamlss(
  frequency ~ .,
  family = PO,
  data = df[, !(names(df) %in% c("log_expected_loss", "year", "month"))]
)
summary(model)

alpha <- 0.99    #alpha 값 수정해야함 ㅜㅜ 
lambda_hat <- fitted(model, "mu")
frequency_VaR <- qPO(p = alpha, mu = lambda_hat)  
summary(frequency_VaR)

windows(width = 7, height = 7)
hist(frequency_VaR)

windows(width = 7, height = 7)
qqnorm(residuals(model))
qqline(residuals(model))


windows(width = 7, height = 7)
plot(lambda_hat, df$frequency,
     xlab = "Predicted", ylab = "Actual",
     main = "Predicted vs Actual")
abline(0, 1, col = "red")
