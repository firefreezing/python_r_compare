
# test Caret package

library(tidyverse)
library(caret)
library(parallel)
library(doParallel)

# read training & testing data
dat_train <- read_csv("./data/clinvar_conflicting_train.csv")
dat_test <- read_csv("./data/clinvar_conflicting_test.csv")

dat_train %<>% mutate(class = factor(class))
dat_test %<>% mutate(class = factor(class))

# # Turn all categorical variables into dummies. 
# dummies_train <- dummyVars(class ~ ., data = dat_train)
# 
# dat_train_new <- predict(dummies_train, newdata = dat_train)
# dat_test_new <- predict(dummies_train, newdata = dat_test)

fitControl <- trainControl(## 10-fold CV
    method = "repeatedcv",
    number = 10,
    ## repeated ten times
    repeats = 2)

start_time <- Sys.time()

n_core <- detectCores()
cl <- makePSOCKcluster(n_core)
registerDoParallel(cl)


set.seed(2046)

gbmFit <- train(class ~ ., 
                data = dat_train, 
                method = "gbm", 
                trControl = fitControl,
                verbose = FALSE,
                metric = "Accuracy")

stopCluster(cl)

end_time <- Sys.time()
difftime(end_time, start_time, units = "mins")

gbmFit

dat_train$class_pred_gbm <- predict(gbmFit, newdata = dat_train)
dat_test$class_pred_gbm <- predict(gbmFit, newdata = dat_test)

dat_test %>%
    mutate(is_pred_right = ifelse(class == class_pred_gbm, 1, 0)) %>%
    summarise(acc = mean(is_pred_right))

# dat_train_var_unq <- dat_train %>%
#     select_if(is.character) %>%
#     gather(key = var, value = value) %>%
#     distinct()
