require(glmnet)

data <- read.csv("data.csv", header=TRUE)
x <- model.matrix(X + Balance ~. * ., data)[, -1]
y <- data$Balance

split_size <- nrow(x) / 2
indexes_train <- sample(seq_len(nrow(x)), split_size)
indexes_test <- -indexes_train

x_train <- x[indexes_train,]
x_test <- x[indexes_test,]
y_train <- y[indexes_train]
y_test <- y[indexes_test]

sequence <- 10^seq(4, -1, length=200)
threshold <- 1e-8

# Linear Regression
linear_model <- lm(y_train~x_train)
linear_predictions <- coef(linear_model)[1]+x_test %*% coef(linear_model)[-1]
linear_model_error <- mean((linear_predictions - y_test)^2)
print(paste0("Linear Performance: ", linear_model_error))

# Ridge
ridge_cross <- cv.glmnet(x_train, y_train, alpha=0, nfolds=5, lambda=sequence, thresh=threshold)
ridge_min <- ridge_cross$lambda.min
ridge_predictions <- predict(ridge_cross, newx=x_test, s=ridge_min)
ridge_model_error <- mean((ridge_predictions - y_test)^2)
print(paste0("Ridge Performance: ", ridge_model_error))

# Lasso
lasso_cross <- cv.glmnet(x_train, y_train, nfolds=5, lambda=sequence, thresh=threshold)
lasso_min <- lasso_cross$lambda.min
lasso_predictions <- predict(lasso_cross, newx=x_test, s=lasso_min)
lasso_model_error <- mean((lasso_predictions - y_test)^2)
print(paste0("Lasso Performance: ", lasso_model_error))

coef_mat <- as.matrix(coef(lasso_cross, lasso_min))
selected <- coef_mat[coef_mat != 0,]
print(paste0("Lasso Matrix: ", length(selected)))

plot(y_test, linear_predictions, ylim=c(-500, 2500), xlab="ground truth", ylab="prediction")
points(y_test, ridge_predictions, col="orange")
points(y_test, lasso_predictions, col="green")