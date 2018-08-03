```{r}
# Step 1: Load, normalize, clean the data and split for modeling
# Load the McKinsey data and replace missing values with NA

mk_train_raw <- read.csv('~/Desktop/Career/McKinsey Analytics Hackathon/train_ZoGVYWq.csv', header=T, na.strings=c(""))
mk_test_raw <- read.csv('~/Desktop/Career/McKinsey Analytics Hackathon/test_66516Ee.csv', header=T, na.strings=c(""))
```

```{r}
# Output the number of missing values for each column

sapply(mk_train_raw,function(x) sum(is.na(x)))
sapply(mk_test_raw,function(x) sum(is.na(x)))
```

```{r}
# Remove 'id', and check the revised dataset

mk_train <- mk_train_raw
mk_test <- mk_test_raw
mk_train$id <- NULL
mk_test$id <- NULL
names(mk_train)
names(mk_test)
```

```{r}
# Normalize the McKinsey train data

mk_train$perc_premium_paid_by_cash_credit <- scale(mk_train$perc_premium_paid_by_cash_credit)[, 1]
mk_train$age_in_days <- scale(mk_train$age_in_days)[, 1]
mk_train$Income <- scale(mk_train$Income)[, 1]
mk_train$Count_3.6_months_late <- scale(mk_train$Count_3.6_months_late)[, 1]
mk_train$Count_6.12_months_late <- scale(mk_train$Count_6.12_months_late)[, 1]
mk_train$Count_more_than_12_months_late <- scale(mk_train$Count_more_than_12_months_late)[, 1]
mk_train$application_underwriting_score <- scale(mk_train$application_underwriting_score)[, 1]
mk_train$no_of_premiums_paid <- scale(mk_train$no_of_premiums_paid)[, 1]
mk_train$premium=scale(mk_train$premium)[, 1]
summary(mk_train)
```

```{r}
# Normalize the test data

mk_test$perc_premium_paid_by_cash_credit <- scale(mk_test$perc_premium_paid_by_cash_credit)[, 1]
mk_test$age_in_days <- scale(mk_test$age_in_days)[, 1]
mk_test$Income <- scale(mk_test$Income)[, 1]
mk_test$Count_3.6_months_late <- scale(mk_test$Count_3.6_months_late)[, 1]
mk_test$Count_6.12_months_late <- scale(mk_test$Count_6.12_months_late)[, 1]
mk_test$Count_more_than_12_months_late <- scale(mk_test$Count_more_than_12_months_late)[, 1]
mk_test$application_underwriting_score <- scale(mk_test$application_underwriting_score)[, 1]
mk_test$no_of_premiums_paid <- scale(mk_test$no_of_premiums_paid)[, 1]
mk_test$premium=scale(mk_test$premium)[, 1]
summary(mk_test)
```

```{r}
# Replace NAs with KNN Imputation, and check

library(lattice)
library(grid)
library(DMwR)
mk_train <- knnImputation(mk_train_raw, k=10) 
mk_test <- knnImputation(mk_test_raw, k=10) 
sapply(mk_train, function(x) sum(is.na(x)))
sapply(mk_test, function(x) sum(is.na(x)))
```

```{r}
# Split the McKinsey train data into trainset and testset, 80 : 20 (random)

set.seed(66)
train <- sample(nrow(mk_train), 0.8*nrow(mk_train), replace = FALSE)
trainset <- mk_train[train,]
testset <- mk_train[-train,]
```

```{r}
# Review the distribution of renewal

prop.table(table(trainset$renewal))
prop.table(table(testset$renewal))
```

```{r}
# Step 2: Build a single model with Logistics Regression
# Get the prediction performance with a single model firstly

# Fit the logistic regression model

logistic <- glm(factor(renewal) ~., family=binomial(link='logit'), data=trainset)
summary(logistic)
```

```{r}
# Predict on testset
# Check accuracy, sensitivity, specificity, and distribution

pred_logstic <- predict(logistic, newdata=testset)
table_pred <- table(testset$renewal, pred_logstic>0.5) ; table_pred
numbers_pred <- as.numeric(table_pred)
TN <- numbers_pred[1]
FP <- numbers_pred[2]
FN <- numbers_pred[3]
TP <- numbers_pred[4]
accuracy_logistic <- (TP+TN)/(TP+FN+FP+TN) ; accuracy_logistic
sensi_logistic <- TP/(TP+FN) ; sensi_logistic
speci_logistic <- TN/(TN+FP) ; speci_logistic
prop.table(table(testset$renewal))
```

```{r}
# Plot the ROC curve

install.packages("pROC")
library(pROC)
plot(roc(testset$renewal, pred_logstic, direction="<"),
     col="yellow", lwd=3, main="prediction accuracy")
```

```{r}
# Step 2 Alternative: Build a single model with Naive Bayes classifier
# Get the prediction performance with a single model firstly

# Fit the Naive Bayes model

library(e1071)
naive_bayes <- naiveBayes(factor(renewal) ~., data=trainset)
naive_bayes
```

```{r}
# Predict on the testset
# Check accuracy, sensitivity, specificity, and distribution

pred_nb <- predict(naive_bayes, newdata=testset)
table_pred <- table(testset$renewal, pred_nb) ; table_pred
numbers_pred <- as.numeric(table_pred)
TN <- numbers_pred[1]
FP <- numbers_pred[2]
FN <- numbers_pred[3]
TP <- numbers_pred[4]
accuracy_nb <- (TP+TN)/(TP+FN+FP+TN) ; accuracy_nb
sensi_nb <- TP/(TP+FN) ; sensi_nb
speci_nb <- TN/(TN+FP) ; speci_nb
prop.table(table(testset$renewal))
```

```{r}
# Train the Naive Bayes model

library(mlr)
# Create a classification learning task and specify the target feature
task <- makeClassifTask(data = testset, target = "renewal")
# Initialize the Naive Bayes classifier
selected_model <- makeLearner("classif.naiveBayes")
# Train the model
train_nb <- train(selected_model, task)
# Read the model learned  
train_nb$learner.model
```

```{r}
# Predict again on the testset without passing the target feature
# Check accuracy, sensitivity, specificity, and distribution

pred_nb <- predict(train_nb, newdata=testset, type='response')
class(testset$renewal)
View(testset$renewal)
table_pred <- table(testset$renewal, pred_nb[[2]]$response) ; table_pred
View(pred_nb)
numbers_pred <- as.numeric(table_pred)
TN <- numbers_pred[1]
FP <- numbers_pred[2]
FN <- numbers_pred[3]
TP <- numbers_pred[4]
accuracy_nb <- (TP+TN)/(TP+FN+FP+TN) ; accuracy_nb
sensi_nb <- TP/(TP+FN) ; sensi_nb
speci_nb <- TN/(TN+FP) ; speci_nb
prop.table(table(testset$renewal))
```

```{r}
# Step 2 Alternative: Build stacking algorithms
# Get and compare the prediction performance with model stacking

# Create submodels

library(caretEnsemble)
library(caret)
library(e1071)
trControl <- trainControl(method='cv', number=5, savePredictions=TRUE, classProbs=TRUE)
modelList <- c('rpart', 'glm', 'knn', 'avNNet')
set.seed(100)
models <- caretList(renewal~., data=trainset, trControl=trControl, methodList=modelList)
results <- resamples(models)
```

```{r}
# Compare the prediction performance of each submodel
# Check the corelations between each submodel

summary(results)
dotplot(results)
modelCor(results)
splom(results)
```

```{r}
# Option 1: Stack using glm

stackControl <- trainControl(method="repeatedcv", number=10, repeats=3, savePredictions=TRUE, classProbs=TRUE)
set.seed(100)
stack.glm <- caretStack(models, method="glm", metric="RMSE", trControl=stackControl)
print(stack.glm)
pred_stacking.glm <- predict(stack.glm, newdata=testset, type="raw")
```

```{r}
# Validate glm weighted accuracy

table_prediction <- table(testset$renewal, pred_stacking.glm>0.5)
table_prediction
numbers_prediction <- as.numeric(table_prediction)
TN=numbers_prediction[1]
FP=numbers_prediction[2]
FN=numbers_prediction[3]
TP=numbers_prediction[4]
accuracy = (TP+TN)/(TP+FN+FP+TN)
sensitivity = TP/(TP+FN)
specificity = TN/(TN+FP)
accuracy
sensitivity
specificity
```

```{r}
# Option 2: Stack using rpart

stackControl <- trainControl(method="repeatedcv", number=10, repeats=3, savePredictions=TRUE, classProbs=TRUE)
set.seed(100)
stack.rpart <- caretStack(models, method="rpart", metric="RMSE", trControl=stackControl)
print(stack.rpart)
pred_stacking.rpart <- predict(stack.rpart, newdata=testset, type="raw")
View(pred_stacking.rpart)
summary(pred_stacking.rpart)
```

```{r}
# Validate rpart weighted accuracy

table_pred <- table(testset$renewal, pred_stacking.rpart>0.5)
table_pred
numbers_prediction <- as.numeric(table_pred)
TN=numbers_prediction[1] ; TN
FP=numbers_prediction[2] ; FP
FN=numbers_prediction[3] ; FN
TP=numbers_prediction[4] ; TP
accuracy = (TP+TN)/(TP+FN+FP+TN)
sensitivity = TP/(TP+FN)
specificity = TN/(TN+FP)
accuracy
sensitivity
specificity
```

```{r}
# Option 3: Stack using knn

stackControl <- trainControl(method="repeatedcv", number=10, repeats=3, savePredictions=TRUE, classProbs=TRUE)
set.seed(100)
stack.knn <- caretStack(models, method="knn", metric="RMSE", trControl=stackControl)
print(stack.knn)
pred_stacking.knn <- predict(stack.knn, newdata=testset, type="raw")
View(pred_stacking.knn)
summary(pred_stacking.knn)
```

```{r}
# Option 3: Stack using knn

table_pred <- table(testset$renewal,pred_stacking.knn>0.5)
table_pred
numbers_prediction <- as.numeric(table_pred)
TN=numbers_prediction[1]
FP=numbers_prediction[2]
FN=numbers_prediction[3]
TP=numbers_prediction[4]
accuracy = (TP+TN)/(TP+FN+FP+TN)
sensitivity = TP/(TP+FN)
specificity = TN/(TN+FP)
accuracy
sensitivity
specificity
```

```{r}
# Option 4: Stack using gbm

stackControl <- trainControl(method="repeatedcv", number=10, repeats=3, savePredictions=TRUE, classProbs=TRUE)
set.seed(100)
stack.gbm <- caretStack(models, method="gbm", metric="RMSE", trControl=stackControl)
print(stack.gbm)
pred_stacking.gbm <- predict(stack.gbm, newdata=testset, type="raw")
```

```{r}
# Validate glm weighted accuracy

table_prediction <- table(testset$renewal, pred_stacking.glm>0.5)
table_prediction
numbers_prediction <- as.numeric(table_prediction)
TN=numbers_prediction[1]
FP=numbers_prediction[2]
FN=numbers_prediction[3]
TP=numbers_prediction[4]
accuracy = (TP+TN)/(TP+FN+FP+TN)
sensitivity = TP/(TP+FN)
specificity = TN/(TN+FP)
accuracy
sensitivity
specificity
```

```{r}
# Step 3: Create calculation functions for insurance net revenue and incentives

cal.revenue <- function(incentive, pred_p, premium){
  # Initialize parameters
  effort=0
  impro_p=0
  revenue=0
  # Equation for the effort-incentives curve
  effort = 10*(1-exp(-incentive/400))
  # Equation for the % improvement in renewal prob vs effort curve
  impro_p = 0.01*20*(1-exp(-effort/5))
  revenue = pred_p*(1+impro_p)*premium-incentive
  return(-revenue)
}
cal.incentive <- function(pred, premium){
  # Initialize parameters
  incentive <- data.frame(sample(x = 1, size = length(pred),replace = TRUE))
  
  for(i in 1:length(pred)){
    if(pred[i] > 0.852564){
       Maxlikely = optim(0.1, pred_p=pred[i], premium = premium[i], cal.revenue, 
                         lower = -0.00000001, upper = -400*log(1+0.5*log((1-((1/pred[i])-1)/0.2))),
                         method="Brent")
    }else{
       Maxlikely = optim(1, pred_p=pred[i], premium=premium[i], cal.revenue,
                         lower = -0.000000001, upper = 10000,
                         method="Brent")
    }
    incentive[i,] = Maxlikely$par
  }
  return(incentive)
}
# why pred[i] > 0.852564
# pred_p*(1+impro_p) < 1
# impro_p < 1/pred_p - 1
# 0.01*20*(1-exp(-(10*(1-exp(-incentive/400)))/5)) < 1/pred_p - 1
# 1-(1/pred_p - 1)/0.2 < exp(-(10*(1-exp(-incentive/400)))/5)
# first limitation
# 1-(1/pred_p - 1)/0.2 > 0 (before using ln on both side)
# pred_p > 5/6
# -0.5*log(1-(1/pred_p - 1)/0.2) > 1-exp(-incentive/400)
# exp(-incentive/400) > 1+0.5*log(1-(1/pred_p - 1)/0.2)
# second limitation
# 1+0.5*log(1-(1/pred_p - 1)/0.2) > 0 (before using ln on both side)
# pred_p > 5/(exp(-2)-6)
```

```{r}
# Predict renewal prob and calculate incentives for McKinsey test data

mk_pred <- predict(logistic, newdata = mk_test, type="response")
mk_incentive <- cal.incentive(mk_pred, mk_test_raw$premium)
summary(mk_incentive)
```

```{r}
# Write csv solution with renewal prob and incentives

library(data.table)
solution <- data.table(`id` = mk_test_raw$id, `renewal` = mk_pred, `incentives` = mk_incentive)
colnames(solution) <- c("id", "renewal", "incentives")
solution[incentives<0,]$incentives=0
write.csv(solution, file = "~/Desktop/Career/McKinsey Analytics Hackathon/solution_logistic regression.csv", row.names = F)
```
