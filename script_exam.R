library(dplyr)
library(magrittr)
library(tidyr)
library(ggplot2)
library(Hmisc)
library(purrr)
library(caret)
library(ranger) 
library(xgboost)
library(kableExtra) # just to make the output nicer
library(mice)
library(stargazer)
library(xtable)
library(pROC) # think we use yardstick instead, nicer plots
library(knitr)
library(grid)
library(yardstick)
library(SciViews)
library(gridExtra)

options(xtable.comment=FALSE)

theme_set(theme_bw()) 

df <- read.csv("pd_data_v2.csv", sep = ";", header = TRUE)

df_corp <- df


# Task 1: Predicion modelling

## Part 1: Inspecting, cleaning and imputing the data set 

# Summary statistics 

caption_at_bottom <- function(expr) {
  x <- capture.output(expr)
  cap <- grep("\\\\caption", x)
  lab <- grep("\\\\label", x)
  last <- grep("\\\\end\\{table", x)
  cat(
    paste(
      c(x[-last], x[cap], x[lab], x[last])[-c(cap, lab)]
    , collapse = "\n")
  , "\n")
}

caption_at_bottom(stargazer(df_corp, summary.stat = c("min","max"),header = FALSE, title  = "Summary statistics"))

# Distribution of defaults and not defaults

df_corp$default <- as.factor(df_corp$default) 

df_corp %>% 
  ggplot(aes(x = default, fill = default)) +
  geom_bar() +
  ggtitle("Distribution of the non-defaulters and defaulters") +
  theme(plot.title = element_text(hjust = 0.5))

# xtable(table(df_corp$default, df_corp$adverse_audit_opinion))

df_corp$adverse_audit_opinion <- 
  ifelse(df_corp$adverse_audit_opinion == 0, yes = 0, no = 1)


# Change classes

df_corp$adverse_audit_opinion <- as.factor(df_corp$adverse_audit_opinion) 
df_corp$industry <- as.factor(df_corp$industry)
df_corp$payment_reminders <- as.factor(df_corp$payment_reminders) 
df_corp$equity <- as.numeric(df_corp$equity)
df_corp$total_assets <- as.numeric(df_corp$total_assets)
df_corp$revenue <- as.numeric(df_corp$revenue)
df_corp$age_of_company <- as.numeric(df_corp$age_of_company)

# Replacing outliers with NA

# First: Take a closer look at the distribution of the values we belive are errors

x <- df_corp %>% 
  select_if(is.numeric)

placeholder <- matrix(ncol=ncol(x), nrow=1)
colnames(placeholder) <- names(x)


for(i in 1:ncol(x)){
  placeholder[,i] <- ifelse(x[,i] > 100000000000 , yes = 1 , 
                            no = ifelse(x[,i] < -100000000000, yes = 1, no = 0)) %>% 
    sum()
}



# Change the obvious errors to NA

df_corp[df_corp < -1000000000000000] <- NA 
df_corp[df_corp >  1000000000000000] <- NA


# We take a new look after replacing the errors with NA

df_corp %>% 
  select(which(sapply(.,class)=="numeric"),default) %>%   
  gather(metric, value, -default) %>% 
  ggplot(aes(x= default, y=value, fill = default))+
  geom_boxplot(show.legend = FALSE) +
  facet_wrap(~ metric, scales = "free")


################## Dealing with the outliers ######################


# Function for replacing outliers with NA's

remove_outliers_na <- function(x) { # To be used on other variables
  qnt <- as.vector(quantile(x, probs=c(0.025, 0.975), na.rm = TRUE))
  y <- x
  y[x < qnt[1]] <- NA
  y[x > qnt[2]] <- NA
  y
}

# Strategy: Replace observarions exceeding treshold values of 0.025 (lower bound) 0.975 (upper bound) with NA. 
# This is done for all numeric variables except from age_of_company, gross_operating_inc_perc, paid_debt_collection and unpaid_debt collection.
# paid and unpaid debt collection will be categorized.

df_corp$profit_margin <- remove_outliers_na(df_corp$profit_margin)
df_corp$operating_margin <- remove_outliers_na(df_corp$operating_margin)
df_corp$EBITDA_margin <- remove_outliers_na(df_corp$EBITDA_margin)
df_corp$interest_coverage_ratio <- remove_outliers_na(df_corp$interest_coverage_ratio)
df_corp$cost_of_debt <- remove_outliers_na(df_corp$cost_of_debt)
df_corp$interest_bearing_debt <- remove_outliers_na(df_corp$interest_bearing_debt)
df_corp$revenue_stability <- remove_outliers_na(df_corp$revenue_stability)
df_corp$equity_ratio <- remove_outliers_na(df_corp$equity_ratio)
df_corp$equity_ratio_stability <- remove_outliers_na(df_corp$equity_ratio_stability)
df_corp$liquidity_ratio_1 <- remove_outliers_na(df_corp$liquidity_ratio_1)
df_corp$liquidity_ratio_2 <- remove_outliers_na(df_corp$liquidity_ratio_2)
df_corp$liquidity_ratio_3 <- remove_outliers_na(df_corp$liquidity_ratio_3)
df_corp$equity <- remove_outliers_na(df_corp$equity)
df_corp$total_assets <- remove_outliers_na(df_corp$total_assets)
df_corp$revenue <- remove_outliers_na(df_corp$revenue)
df_corp$amount_unpaid_debt <- remove_outliers_na(df_corp$amount_unpaid_debt)

df_corp$default <- recode_factor(
  df_corp$default,
  `0` = "No",
  `1` = "Yes")

df_corp <- df_corp %>% 
  select(-equity_ratio_stability) # Removed due to collinearity and if look at the earlier plots it make sense.

# Imputing NAs. 

# temp_imputed1 <- mice(df_corp, m=2,maxit=3,meth='mean',seed=500)

# temp_imputed2 <- mice(df_corp, m=2,maxit=3,meth='pmm',seed=500)

# saveRDS(temp_imputed1, file = "imputeMEAN.Rdata")

# saveRDS(temp_imputed2, file = "impute.Rdata") 

# densityplot(temp_imputed1, drop.unused.levels = TRUE) # Save plots again

# densityplot(temp_imputed2, drop.unused.levels = TRUE) # Save plots again

# Running the imputations and generating density plots of the imputed values is higly time consuming. Therefore, we have saved these files and will upload them in chunk xx. 

# See appendix for plots. 

# Uploading imputed values

temp_imputed1 <- readRDS(file = "imputeMEAN.Rdata")
temp_imputed2 <- readRDS(file = "impute.Rdata")

# We complete the impution and check the result

complete_imputed <- complete(temp_imputed2, 1)

Orginal <- dim(df_corp)
Imputed <- dim(complete_imputed)

df_cleaned <- complete_imputed

# Generating dummies for paid and unpaid debt collection

df_cleaned$unpaid_debt_collection <- ifelse(df_cleaned$unpaid_debt_collection > 0, yes = 1, no = 0) # 1 = Companies that have unpaid debt collection

df_cleaned$paid_debt_collection <- ifelse(df_cleaned$paid_debt_collection > 0, yes = 1, no = 0) # 1 = Companies that have previously had debt collection

df_cleaned$paid_debt_collection <- as.factor(df_cleaned$paid_debt_collection)
df_cleaned$unpaid_debt_collection <- as.factor(df_cleaned$unpaid_debt_collection)


# Density plot after cleaning

df_cleaned %>% 
  select_if(is.numeric) %>% 
  gather(metric, value) %>% 
  ggplot(aes(value, fill = metric))+
  geom_density(show.legend = FALSE) +
  facet_wrap(~ metric, scales = "free")

# Boxplot after cleaning

df_cleaned %>% 
  select(which(sapply(.,class)=="numeric"),default) %>%   
  gather(metric, value, -default) %>% 
  ggplot(aes(x= default, y=value, fill = default))+
  geom_boxplot(show.legend = FALSE) +
  facet_wrap(~ metric, scales = "free")

##Prediction modelling


### Preparations

# For all models we will remove the equity, total_assets, revenue and amount_unpaid_dept because we want to work with relative values.

df_reduced <- df_cleaned %>% 
  select(-equity, -total_assets, -revenue, -amount_unpaid_debt)

# Last thing we do before we split the data and start modelling is checking for mulitcollinearity.
# To do this we will be using vic function in the car package.

library(car)

mc_vif_1 <- vif(glm(default ~ ., data = df_reduced, family = "binomial"))

mc_vif_1

# We see that for the glm we should remove operating margin

mc_vif_2 <- vif(glm(default ~ .-operating_margin, data = df_reduced, family = "binomial"))

mc_vif_2 # By checking again all looks okay

# Stepwise

# stepwise <- step(glm(default ~ .-operating_margin, data = df_reduced, family = "binomial"), direction="both")

# Removed after stepwise

mc_vif_3 <- vif(glm(default ~ .-operating_margin - paid_debt_collection - liquidity_ratio_1 - profit_margin, data = df_reduced, family = "binomial"))

mc_vif_3

# Split data in to train/test

set.seed(1)

index <- createDataPartition(df_reduced$default, p = 0.7, list = FALSE)
train_data <- df_reduced[index, ]
test_data <- df_reduced[-index, ]

`Train defaults`<- summary(train_data$default)[2]/nrow(train_data)

`Test defaults`<- summary(test_data$default)[2]/nrow(test_data)

xtable(rbind(`Train defaults`,`Test defaults`))


############# Making some models ###################

ctrl <- trainControl(method = "repeatedcv", 
                     number = 10, 
                     repeats =10, 
                     verboseIter = FALSE,
                     sampling = "smote",
                     classProbs=TRUE)


#################### Logistic Regression Model #############################3


library(MLmetrics)
# After we have runned the model we see that industry and gross have some insignificant values,
# we then remove grossbased on AIC and significant value of the variable

set.seed(1)

model_glm <- train(default ~ .- operating_margin 
                   - paid_debt_collection 
                   - liquidity_ratio_1 
                   - profit_margin
                   - gross_operating_inc_perc,
                   data = train_data,
                   method = "glm",
                   trControl = ctrl,
                   metric = "AUC")

# Threshold 0.3

summary_glm <- xtable(summary(model_glm), caption = "Summary glm model")

print(summary_glm, scalebox=1, caption.placement = "bottom", label = "tab:sumtable")

plot(varImp(model_glm), main = "Variables Importance GLM")

glm_pred <- data.frame(actual = test_data$default,
                       predict(model_glm, newdata = test_data, type = "prob"))

glm_pred$predict <- ifelse(glm_pred$Yes > 0.4, "Yes", "No")
glm_pred$predict <- as.factor(glm_pred$predict)

cm_glm <-confusionMatrix(glm_pred$predict, test_data$default, positive= "Yes")

confusion_glm <- xtable(as.matrix(cm_glm$table), caption = "Confusion matrix GLM")

# print(confusion_glm, scalebox = 1, caption.placement = "bottom", label = "tab:confusion_glm")

perf_glm_3 <- as.matrix(cm_glm$byClass)
colnames(perf_glm_3) <- c("GLM 0.3")

performance_glm_3 <- xtable(perf_glm_3, caption = "Performance indicators GLM")

# print(performance_glm_5, scalebox = 1, caption.placement = "bottom", label = "tab:performance_glm_3")

# Threshold 0.4

glm_pred <- data.frame(actual = test_data$default,
                       predict(model_glm, newdata = test_data, type = "prob"))

glm_pred$predict <- ifelse(glm_pred$Yes > 0.4, "Yes", "No")
glm_pred$predict <- as.factor(glm_pred$predict)

cm_glm <-confusionMatrix(glm_pred$predict, test_data$default, positive= "Yes")

confusion_glm <- xtable(as.matrix(cm_glm$table), caption = "Confusion matrix GLM threshold: 0.4")

print(confusion_glm, scalebox = 1, caption.placement = "bottom", label = "tab:confusion_glm")

perf_glm_4 <- as.matrix(cm_glm$byClass)
colnames(perf_glm_4) <- c("GLM 0.4")

performance_glm_4 <- xtable(perf_glm_4, caption = "Performance indicators GLM")

# print(performance_glm_4, scalebox = 1, caption.placement = "bottom", label = "tab:performance_glm_4")

# Treshold 0.5

glm_pred <- data.frame(actual = test_data$default,
                       predict(model_glm, newdata = test_data, type = "prob"))

glm_pred$predict <- ifelse(glm_pred$Yes > 0.5, "Yes", "No")
glm_pred$predict <- as.factor(glm_pred$predict)

cm_glm <-confusionMatrix(glm_pred$predict, test_data$default, positive= "Yes")

confusion_glm <- xtable(as.matrix(cm_glm$table), caption = "Confusion matrix GLM")

# print(confusion_glm, scalebox = 1, caption.placement = "bottom", label = "tab:confusion_glm")

perf_glm_5 <- as.matrix(cm_glm$byClass)
colnames(perf_glm_5) <- c("GLM 0.5")

performance_glm_5 <- xtable(perf_glm_5, caption = "Performance indicators GLM")

# print(performance_glm_5, scalebox = 1, caption.placement = "bottom", label = "tab:performance_glm")

# Comparing

comp_thresh_glm <- cbind(perf_glm_3, perf_glm_4, perf_glm_5)

print(xtable(comp_thresh_glm, caption = "Performance indicators GLM"), scalebox = 1, caption.placement = "bottom", label = "tab:comp_thresh_glm")

# ROC curve glm

glm_predictions <- predict(model_glm, test_data, type="prob")

test_glm <- test_data %>%
  select(default) %>% 
  mutate(glm_prob_predictions = glm_predictions$Yes)

glm_auc <- test_glm %>% 
  roc_auc(default, glm_prob_predictions)

glm_auc <- paste(round(glm_auc$.estimate, 4), sep = "")

test_glm %>%
  roc_curve(default, glm_prob_predictions) %>% 
  ggplot(aes(x = 1- specificity, y = sensitivity)) +
  geom_path() +
  geom_abline(lty = 3) +
  annotate("text",x = 0.7, y = 0.4, label = glm_auc) +
  coord_equal() +
  ggtitle("ROC curve GLM (Logistic)") +
  theme(plot.title = element_text(hjust = 0.5))



############################ Model 2: Random Forest ######################################

tunegrid <- expand.grid(.mtry= 4,
                        .splitrule = "gini",
                        .min.node.size = 10) # tuning with mtry = 1:5, min.node.size = 10,20

set.seed(1)

model_rf <- caret::train(default ~ .,
                         data = train_data,
                         method = "ranger",
                         trControl = ctrl,
                         tuneGrid = tunegrid,
                         num.trees = 100,
                         importance = "permutation")
                        

model_rf <- readRDS("rf.Rdata")

# Treshold 0.3

rf_pred <- data.frame(actual = test_data$default,
                       predict(model_rf, newdata = test_data, type = "prob"))

rf_pred$predict <- ifelse(rf_pred$Yes > 0.3, "Yes", "No")
rf_pred$predict <- as.factor(rf_pred$predict)

plot(varImp(model_rf),main = "Variables Importance Random Forest")

cm_rf <- confusionMatrix(rf_pred$predict, test_data$default, positive = "Yes")

confusion_rf <- xtable(as.matrix(cm_rf$table), caption = "Confusion matrix Random Forest")

# print(confusion_rf, scalebox = 1, caption.placement = "bottom", label = "tab:confusion_rf")

perf_rf_3 <- as.matrix(cm_rf$byClass)
colnames(perf_rf_3) <- c("RF 0.3")

performance_rf_3 <- xtable(perf_rf_3, caption = "Performance indicators Random Forest")

# print(performance_rf_3, scalebox = 1, caption.placement= "bottom", label = "tab:performance_rf_3")

# Treshold 0.4

rf_pred <- data.frame(actual = test_data$default,
                       predict(model_rf, newdata = test_data, type = "prob"))

rf_pred$predict <- ifelse(rf_pred$Yes > 0.4, "Yes", "No")
rf_pred$predict <- as.factor(rf_pred$predict)

cm_rf <- confusionMatrix(rf_pred$predict, test_data$default, positive = "Yes")

confusion_rf <- xtable(as.matrix(cm_rf$table), caption = "Confusion matrix Random Forest threshold: 0.4")

print(confusion_rf, scalebox = 1, caption.placement = "bottom", label = "tab:confusion_rf")

perf_rf_4 <- as.matrix(cm_rf$byClass)
colnames(perf_rf_4) <- c("RF 0.4")

performance_rf_4 <- xtable(perf_rf_4, caption = "Performance indicators Random Forest")

# print(performance_rf_4, scalebox = 1, caption.placement= "bottom", label = "tab:performance_rf_4")

# Treshold 0.5

rf_pred <- data.frame(actual = test_data$default,
                       predict(model_rf, newdata = test_data, type = "prob"))

rf_pred$predict <- ifelse(rf_pred$Yes > 0.5, "Yes", "No")
rf_pred$predict <- as.factor(rf_pred$predict)

cm_rf <- confusionMatrix(rf_pred$predict, test_data$default, positive = "Yes")

confusion_rf <- xtable(as.matrix(cm_rf$table), caption = "Confusion matrix Random Forest")

# print(confusion_rf, scalebox = 1, caption.placement = "bottom", label = "tab:confusion_rf")

perf_rf_5 <- as.matrix(cm_rf$byClass)
colnames(perf_rf_5) <- c("RF 0.5")

performance_rf_5 <- xtable(perf_rf_5, caption = "Performance indicators Random Forest")

# print(performance_rf_5, scalebox = 1, caption.placement= "bottom", label = "tab:performance_rf_5")

# Comparing

comp_thresh_rf <- cbind(perf_rf_3, perf_rf_4, perf_rf_5)

print(xtable(comp_thresh_rf, caption = "Performance indicators Rf"), scalebox = 1, caption.placement = "bottom", label = "tab:comp_thresh_rf")

# ROC curve rf

rf_predictions <- predict(model_rf, test_data, type = "prob")

test_rf <- test_data %>%
  select(default) %>% 
  mutate(rf_prob_predictions = rf_predictions$Yes)

rf_auc <- test_rf %>% 
              roc_auc(default, rf_prob_predictions)

rf_auc <- paste(round(rf_auc$.estimate, 4), sep = "")

test_rf %>%
  roc_curve(default, rf_prob_predictions) %>% 
  ggplot(aes(x = 1- specificity, y = sensitivity)) +
  geom_path() +
  geom_abline(lty = 3) +
  annotate("text",x = 0.7, y = 0.4, label = rf_auc) +
  coord_equal() +
  ggtitle("ROC curve Random Forest (Ranger)") +
  theme(plot.title = element_text(hjust = 0.5))



############################################ Model 3: Xgboost

xgb_grid <- expand.grid(nrounds = 300,
                        max_depth = 7, 
                        min_child_weight = 1,
                        subsample = 1,
                        gamma = .1,
                        colsample_bytree = 0.8,
                        eta = .1)

set.seed(1)

model_xgb <- caret::train(default ~ .,
                          data = train_data,
                          method = "xgbTree",
                          tuneGrid =xgb_grid,
                          trControl = ctrl)



# threshold 0.3

plot(varImp(model_xgb),main = "Variables Importance Xgboost")

xgb_pred <- data.frame(actual = test_data$default,
                      predict(model_xgb, newdata = test_data, type = "prob"))

xgb_pred$predict <- ifelse(xgb_pred$Yes > 0.2, "Yes", "No")
xgb_pred$predict <- as.factor(xgb_pred$predict)

cm_xgb <- confusionMatrix(xgb_pred$predict, xgb_pred$actual, positive = "Yes")

confusion_xgb <- xtable(as.matrix(cm_xgb$table), caption = "Confusion matrix xgboost treshhold: 0.3")

print(confusion_xgb, scalebox = 1, caption.placement = "bottom", label = "tab:confusion_xgb")

perf_xgb_3 <- as.matrix(cm_xgb$byClass)
colnames(perf_xgb_3) <- c("XGB 0.3")

performance_xgb_3 <- xtable(perf_xgb_3, caption = "Performance indicators xgboost")

print(performance_xgb_3, scalebox = 1, caption.placement= "bottom", label = "tab:performance_xgb_3")

# threshold 0.4

xgb_pred <- data.frame(actual = test_data$default,
                      predict(model_xgb, newdata = test_data, type = "prob"))

xgb_pred$predict <- ifelse(xgb_pred$Yes > 0.4, "Yes", "No")
xgb_pred$predict <- as.factor(xgb_pred$predict)

cm_xgb <- confusionMatrix(xgb_pred$predict, xgb_pred$actual, positive = "Yes")

# confusion_xgb <- xtable(as.matrix(cm_xgb$table), caption = "Confusion matrix xgboost treshhold: 0.4")

print(confusion_xgb, scalebox = 1, caption.placement = "bottom", label = "tab:confusion_xgb")

perf_xgb_4 <- as.matrix(cm_xgb$byClass)
colnames(perf_xgb_4) <- c("XGB 0.4")

performance_xgb_4 <- xtable(perf_xgb_4, caption = "Performance indicators xgboost")

# print(performance_xgb_4, scalebox = 1, caption.placement= "bottom", label = "tab:performance_xgb_4")

# threshold 0.5

xgb_pred <- data.frame(actual = test_data$default,
                      predict(model_xgb, newdata = test_data, type = "prob"))

xgb_pred$predict <- ifelse(xgb_pred$Yes > 0.5, "Yes", "No")
xgb_pred$predict <- as.factor(xgb_pred$predict)

cm_xgb <- confusionMatrix(xgb_pred$predict, xgb_pred$actual, positive = "Yes")

confusion_xgb <- xtable(as.matrix(cm_xgb$table), caption = "Confusion matrix xgboost treshhold: 0.5")

# print(confusion_xgb, scalebox = 1, caption.placement = "bottom", label = "tab:confusion_xgb")

perf_xgb_5 <- as.matrix(cm_xgb$byClass)
colnames(perf_xgb_5) <- c("XGB 0.5")

performance_xgb_5 <- xtable(perf_xgb_5, caption = "Performance indicators xgboost")

# print(performance_xgb_5, scalebox = 1, caption.placement= "bottom", label = "tab:performance_xgb_5")

# Comparing

comp_thresh_xgb <- cbind(perf_xgb_3, perf_xgb_4, perf_xgb_5)

print(xtable(comp_thresh_xgb, caption = "Performance indicators Xgb"), scalebox = 1, caption.placement = "bottom", label = "tab:comp_thresh_Xgb")

# ROC curve xgb

xgb_predictions <- predict(model_xgb, test_data, type = "prob")

test_xgb <- test_data %>%
  select(default) %>% 
  mutate(xgb_prob_predictions = xgb_predictions$Yes)

xgb_auc <- test_xgb %>% 
              roc_auc(default, xgb_prob_predictions)

xgb_auc <- paste(round(xgb_auc$.estimate, 4), sep = "")

test_xgb %>%
  roc_curve(default, xgb_prob_predictions) %>% 
  ggplot(aes(x = 1- specificity, y = sensitivity)) +
  geom_path() +
  geom_abline(lty = 3) +
  annotate("text",x = 0.7, y = 0.4, label = xgb_auc) +
  coord_equal() +
  ggtitle("ROC curve Xgboost") +
  theme(plot.title = element_text(hjust = 0.5))

######################################## Comparing the models

table_all <- cbind(perf_glm_4,perf_rf_4,perf_xgb_3)

performance_all <- xtable(table_all, caption = "Performance indicators all models")
print(performance_all, scalebox = 1, caption.placement= "bottom", label = "tab:performance_all")

models <- list(glm = model_glm,
               rf = model_rf,
               xgb = model_xgb)

resampling <- resamples(models)

bwplot(resampling, main = "Comparing of the models")

comp_table <- summary(resampling) 
  
caption_at_bottom(stargazer(comp_table$statistics$Accuracy, header = FALSE, title = "Accuracy" ))
caption_at_bottom(stargazer(comp_table$statistics$Kappa, header = FALSE, title = "Kappa"))


# Assignment 5

## 5.2. RWA calculation

#define values of some constants

RWA <- function(x){
   
LGD = 0.45
EAD = 100
M = 2.5
S = 50
  
Ra <- 1-exp(-50*x)
Rb <- 1-exp(-50) 

Re <- S-5
Rf <- 0.04 * (1-(Re/45))
R <- 0.12 * (Ra/Rb) + 0.24 * (1-Ra/Rb) - Rf

Ba <- 0.11852 - 0.05478*ln(x)

B <- Ba^2

Ka <- (1-R)^(-0.5)
Kb <- qnorm(x)
Kc <- (R/(1-R))^0.5
Kd <- qnorm(0.999)
Ke <- x*LGD
Kf <- (1-1.5*B)^(-1)
Kg <- 1+((M-2.5)*B)
  
K <- (LGD*pnorm(Ka*Kb + Kc*Kd) - Ke)*Kf*Kg

RWA = K*12.5*EAD

RWA  
}

RWA_glm <- RWA(glm_predictions$Yes)
RWA_rf <- RWA(rf_predictions$Yes)
RWA_xgb <- RWA(xgb_predictions$Yes)


#Compare the RWA in each model

RWA_models <- as.data.frame(cbind(RWA_glm, RWA_rf, RWA_xgb))

pd_df <- cbind(glm_predictions$Yes, rf_predictions$Yes, xgb_predictions$Yes)
colnames(pd_df) <- c("GLM", "RF", "XGB")

caption_at_bottom(stargazer(pd_df, header = FALSE, title  = "PD distribution"))

caption_at_bottom(stargazer(RWA_models, header = FALSE, title  = "RWA comparison"))


\[
  \makebox[\linewidth]{$\displaystyle
    \begin{aligned} R = 0.12\cdot\frac{1-exp(-50 \cdot PD)}{1-exp(-50)}+0.24 \cdot [1-\frac{1-exp(-50 \cdot PD)}{1-exp(-50)}]-0.04 \cdot (1-\frac{(s-5)}{45})\end{aligned}
  $}
\]

\[
  \makebox[\linewidth]{$\displaystyle
    \begin{aligned}MA = (0.11852-0.05478 \cdot ln(PF))^2\end{aligned}
  $}
\]

\[
  \makebox[\linewidth]{$\displaystyle
    \begin{aligned}CR = [LGD \cdot N [(1-R)^{-0.5} \cdot G(PD) + (\frac{R}{1-R})^{0.5}\cdot G(0.999)]-PD \cdot LGD]\cdot (1-1.5 \cdot b)^{-1}\cdot(1+(M-2.5)\cdot b)\end{aligned}
  $}
\]

\[
  \makebox[\linewidth]{$\displaystyle
    \begin{aligned}RWA = K \cdot 12.5 \cdot EAD\end{aligned}
  $}
\]


p_glm <- ggplot(RWA_models, aes(x= 1:nrow(RWA_models))) + 
            geom_line(aes(y = RWA_glm), color = "steelblue") +
            ggtitle("GLM logistic") +
            theme(plot.title = element_text(hjust = 0.5)) +
            ylab("RWA") +
            xlab("")


p_rf <- ggplot(RWA_models, aes(x= 1:nrow(RWA_models))) +
            geom_line(aes(y = RWA_rf), color="darkgreen") +
            ggtitle("Random Forest") +
            theme(plot.title = element_text(hjust = 0.5)) +
            ylab("RWA") +
            xlab("")
    
            
p_xgb <- ggplot(RWA_models, aes(x= 1:nrow(RWA_models))) +          
            geom_line(aes(y = RWA_xgb), color ="darkred") +
            ggtitle("xgBoost") +
            theme(plot.title = element_text(hjust = 0.5)) +
            ylab("RWA") +
            xlab("")
     
grid.arrange(p_glm, p_rf, p_xgb, nrow = 3)


# Appendix


### load images 

include_graphics('imputation_pmm.jpeg')


### load images 

include_graphics('imputation_mean.jpeg',)


### load images 

include_graphics('rf_tuning_plot.jpeg',)

### load images 

include_graphics('xgb_tuning_plot.jpeg',)
