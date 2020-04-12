Sys.setenv(LANG = "en")

# Data processing library
library(data.table)       # Data manipulation
library(plyr)             # Data manipulation
library(stringr)          # String, text processing
library(vita)             # Quickly check variable importance
library(dataPreparation)  # Data preparation library
library(woeBinning)       # Decision tree-based binning for numerical and categorical variables
library(Boruta)           # Variable selection

# Machine learning library
library(mlr)          # Machine learning framework
library(caret)         # Data processing and machine learning framework
library(MASS)          # LDA
library(randomForest)  # RF
library(gbm)           # Boosting Tree
library(xgboost)       # XGboost
library(LLM)           # Logit Leaf Model
library(dplyr)
library(tidyr)


install.packages("")
#################################Load the table##############################################

setwd("C:/Users/xzong/Desktop/MBD/Machine Learning/Group Project")
train                       = read.csv("train.csv", sep=",")
test                        = read.csv("test.csv",sep = ",")

train_backup <- train
test_backup <- test
set.seed(2)


#We randomly selected 1000000 rows for our training set and test set

train_p <- train[sample(nrow(train), 1000000), ]
test_p <- test[sample(nrow(test),1000000), ]

#View(train_p)

###################################Data Processing and Feature Engineering##############################################

#Define the function of feature selectoin

#Reference: Course of Statistical & machine learning approches for marketing from IESEG SCHOOL OF MANAGEMENT. Professor: Minh Phan  

FisherScore <- function(basetable, depvar, IV_list) {
  "
  This function calculate the Fisher score of a variable.
  
  Ref:
  ---
  Verbeke, W., Dejaeger, K., Martens, D., Hur, J., & Baesens, B. (2012). New insights into churn prediction in the telecommunication sector: A profit driven data mining approach. European Journal of Operational Research, 218(1), 211-229.
  "
  
  # Get the unique values of dependent variable
  DV <- unique(basetable[, depvar])
  
  IV_FisherScore <- c()
  
  for (v in IV_list) {
    fs <- abs((mean(basetable[which(basetable[, depvar]==DV[1]), v]) - mean(basetable[which(basetable[, depvar]==DV[2]), v]))) /
      sqrt((var(basetable[which(basetable[, depvar]==DV[1]), v]) + var(basetable[which(basetable[, depvar]==DV[2]), v])))
    IV_FisherScore <- c(IV_FisherScore, fs)
  }
  
  return(data.frame(IV=IV_list, fisher_score=IV_FisherScore))
}



varSelectionFisher <- function(basetable, depvar, IV_list, num_select=20) {
  "
  This function will calculate the Fisher score for all IVs and select the best
  top IVs.

  Assumption: all variables of input dataset are converted into numeric type.
  "
  
  fs <- FisherScore(basetable, depvar, IV_list)  # Calculate Fisher Score for all IVs
  num_select <- min(num_select, ncol(basetable))  # Top N IVs to be selected
  return(as.vector(fs[order(fs$fisher_score, decreasing=T), ][1:num_select, 'IV']))
  
}


#train.raw <- train_p[sample(nrow(train_p), nrow(train_p) * .50),]
train.raw <- train_p
test.raw <- test_p
test.raw$HasDetections<- NA
data.raw <- rbind(train.raw, test.raw)

rm(train.raw)
rm(test.raw)
gc()


#Reference Link: https://www.kaggle.com/rintaromasuda/logistic-regression-with-basic-feature-engineering. Rintaro Masuda

#Factor all of the columns except for the columns in the cols.keep

cols.keep <- c("Census_PrimaryDiskTotalCapacity",
               "Census_SystemVolumeTotalCapacity",
               "MachineIdentifier",
               "HasDetections")
for(colName in colnames(data.raw)) {
  if(!(colName %in% cols.keep)) {
    data.raw[[colName]] <- as.factor(data.raw[[colName]])
  }
}

#Categorize the Census_PrimaryDiskTotalCapacity column.

data.raw[["Census_PrimaryDiskTotalCapacityCategory"]] <-
  cut(data.raw[["Census_PrimaryDiskTotalCapacity"]],
      c(0, 1024*32, 1024*64, 1024*128, Inf))


#Categorize the EngineVersion column.

temp <- substr(data.raw[["EngineVersion"]], 5, length(data.raw[["EngineVersion"]]))
temp <- substr(temp, 1, regexpr("\\.", temp) - 1)
temp <- substr(temp, 1, 2)
temp <- ifelse(temp %in% c("12",
                           "13",
                           "14",
                           "15"),
               temp,
               "other")
data.raw[["EngineVersionCategory"]] <- as.factor(temp)

#Categorize the AppVersion column.

temp <- substr(data.raw[["AppVersion"]], 1, 4)
data.raw[["AppVersionCategory"]] <- as.factor(temp)

#Categorize the SmartScreen column.

temp <- tolower(data.raw[["SmartScreen"]])
temp <- gsub("requiredadmin", "requireadmin", temp)
temp <- gsub("promt", "prompt", temp)
temp <- gsub("promprt", "prompt", temp)
temp <- gsub("deny", "", temp)
temp <- gsub("of$", "off", temp)
temp <- ifelse(temp %in% c("block",
                           "existsnotset",
                           "off",
                           "on",
                           "prompt",
                           "requireadmin",
                           "warn"),
               temp,
               "other")
data.raw[["SmartScreenCategory"]] <- as.factor(temp)

#Categorize the AVProductStatesIdentifier column.

temp <- as.integer(as.character(data.raw[["AVProductStatesIdentifier"]]))
temp <- cut(temp, seq(1, 80001, by = 10000))
data.raw[["AVProductStatesIdentifierCategory"]] <- temp


#Categorize the IeVerIdentifierCategory column.

temp <- as.integer(as.character(data.raw[["IeVerIdentifier"]]))
temp <- cut(temp, seq(1, 501, by = 50))
data.raw[["IeVerIdentifierCategory"]] <- as.factor(temp)


#Factor all of the columns except for the columns in the cols.keep and fill nas with "Missing".

factorCols <- unlist(lapply(data.raw, is.factor))
for(colName in colnames(data.raw[, factorCols])) {
  if(!(colName %in% cols.keep)) {
    data.raw[[colName]] <- factor(ifelse(!is.na(data.raw[[colName]]),
                                         as.character(data.raw[[colName]]),
                                         "Missing"))
  }
}



dv_list <- c('HasDetections')
# Independent variable (IV)
iv_list <- setdiff(colnames(data.raw), dv_list)  # Exclude the target variable
iv_list <- setdiff(iv_list, 'MachineIdentifier')  # Exclude the client_id


#Check the class and missing value of each column
for (v in iv_list) {
  print(v)
  print(sum(is.na(data.raw[,v])))
  print(class(data.raw[,v]))
  print("--------")
}


#Convert all of the columns into integer except for the columns in the cols.keep

for (f in iv_list) {
  if(!(f %in% cols.keep)){
    levels <- sort(unique(c(data.raw[[f]])))
    data.raw[[f]] <- as.integer(data.raw[[f]], levels = levels)+1
  }
}


for (f in iv_list) {
  print(f)
  print(sum(is.na(data.raw[[f]])))
}



####################################Modeling#####################################

data.raw <- as.data.table(data.raw)

train.raw <- data.raw[!is.na(HasDetections),]
test.raw <- data.raw[is.na(HasDetections),]

train.raw[, MachineIdentifier := NULL]
test.raw[, HasDetections := NULL]

#View(test.raw)

# Calculate Fisher Score for all variable
# Get the IV and DV list

train <- as.data.frame(train.raw)
test <- as.data.frame(test.raw)

train$Census_PrimaryDiskTotalCapacity <- NULL
train$Census_SystemVolumeTotalCapacity <- NULL

#View(train)
write.csv(train, file = "C:/Users/xzong/Desktop/MBD/Machine Learning/Group Project/base_table.csv")


dv_list <- c('HasDetections')
# Independent variable (IV)
iv_list <- setdiff(colnames(train), dv_list)  # Exclude the target variable
iv_list <- setdiff(iv_list, 'MachineIdentifier')  # Exclude the client_id

#View(train)

fs <- FisherScore(train, dv_list, iv_list)
View(fs)

# Select top 20 variables according to the Fisher Score
best_fs_var <- varSelectionFisher(train, dv_list, iv_list, num_select=20)
head(best_fs_var, 10)





#########################Split the data set for validation and training set##################

# Apply variable selection to the data
# Train
var_select <- names(train)[names(train) %in% best_fs_var]
train_processed <- train[, c(var_select, 'HasDetections')]



train_idx <- caret::createDataPartition(y=train_processed[, 'HasDetections'], p=.6, list=F)
train_final <- train_processed[train_idx, ]  # Train 60%
valid_test <- train_processed[-train_idx, ]  # Valid + Test 40%


valid_idx <- caret::createDataPartition(y=valid_test[, 'HasDetections'], p=.5, list=F)
valid <- valid_test[valid_idx, ]  # Valid 20%
test <- valid_test[-valid_idx, ]  # Test 20%



#############Logistic Regression

rdesc = makeResampleDesc("CV", iters=5, predict="both")

# Define the model
learner <- makeLearner("classif.logreg", predict.type="prob", fix.factors.prediction=T)

# Define the task
train_task <- makeClassifTask(id="train", data=train_final, target="HasDetections")

# Set hyper parameter tuning
tune_params <- makeParamSet(
)
ctrl = makeTuneControlGrid()

# Run the hyper parameter tuning with k-fold CV
if (length(tune_params$pars) > 0) {
  # Run parameter tuning
  res <- tuneParams(learner, task=train_task, resampling=rdesc,
                    par.set=tune_params, control=ctrl, measures=list(mlr::auc))
  
  # Extract best model
  best_learner <- res$learner
  
} else {
  # Simple cross-validation
  res <- resample(learner, train_task, rdesc, measures=list(mlr::auc, setAggregation(mlr::auc, train.mean)))
  
  # No parameter for tuning, only 1 best learner
  best_learner <- learner
}


best_md <- mlr::train(best_learner, train_task)

# Make prediction on valid data
pred <- predict(best_md, newdata=valid)
performance(pred, measures=mlr::auc)

# Make prediction on test data
pred <- predict(best_md, newdata=test)
performance(pred, measures=mlr::auc)


############Random Forest

rdesc = makeResampleDesc("CV", iters=5)

# Define the model
learner <- makeLearner("classif.randomForest", predict.type="prob", fix.factors.prediction=T)

# Define the task
train_task <- makeClassifTask(id="train", data=train_final, target="HasDetections")

# Set hyper parameter tuning
tune_params <- makeParamSet(
  makeDiscreteParam('ntree', value=c(100, 250, 500, 750, 1000)),
  makeDiscreteParam('mtry', value=round(sqrt((ncol(train_processed)-1) * c(0.1, 0.25, 0.5, 1, 2, 4))))
)


ctrl = makeTuneControlGrid()

# Run the hyper parameter tuning with k-fold CV
if (length(tune_params$pars) > 0) {
  # Run parameter tuning
  res <- tuneParams(learner, task=train_task, resampling=rdesc,
                    par.set=tune_params, control=ctrl, measures=list(mlr::auc))
  
  # Extract best model
  best_learner <- res$learner
  
} else {
  # Simple cross-validation
  res <- resample(learner, train_task, rdesc, measures=list(mlr::auc))
  
  # No parameter for tuning, only 1 best learner
  best_learner <- learner
}

best_md <- mlr::train(best_learner, train_task)

# Make prediction on valid data
pred <- predict(best_md, newdata=valid)
performance(pred, measures=mlr::auc)

# Make prediction on test data
pred <- predict(best_md, newdata=test)
performance(pred, measures=mlr::auc)

