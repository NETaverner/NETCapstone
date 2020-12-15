#Libraries
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(corrplot)) install.packages("recorrplotadr", repos = "http://cran.us.r-project.org")
if(!require(gridExtra)) install.packages("gridExtra", repos = "http://cran.us.r-project.org")
if(!require(pROC)) install.packages("pROC", repos = "http://cran.us.r-project.org")
if(!require(caTools)) install.packages("caTools", repos = "http://cran.us.r-project.org")
if(!require(caretEnsemble)) install.packages("caretEnsemble", repos = "http://cran.us.r-project.org")
if(!require(grid)) install.packages("grid", repos = "http://cran.us.r-project.org")
if(!require(readr)) install.packages("readr", repos = "http://cran.us.r-project.org")
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(ggfortify)) install.packages("ggfortify", repos = "http://cran.us.r-project.org")
if(!require(glmnet)) install.packages("glmnet", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(nnet)) install.packages("nnet", repos = "http://cran.us.r-project.org")
if(!require(funModeling)) install.packages("funModeling", repos = "http://cran.us.r-project.org")
#if(!require(Momocs)) install.packages("Momocs", repos = "http://cran.us.r-project.org")
#if(!require(klaR)) install.packages("klaR", repos = "http://cran.us.r-project.org")
if(!require(RefManageR)) install.packages("RefManageR", repos = "http://cran.us.r-project.org")
if(!require(kableExtra)) install.packages("kableExtra", repos = "http://cran.us.r-project.org")


# The data file will be loaded from my personal computer
data <- read.csv("/Users/user/Documents/DataScience/FinalCapstone/NETCapstone/DataFiles/column_2C_weka.csv")

# attributes 
#Class

# 6 Features 


#Data Analysis

str(data)

head(data)

summary(data)

#check for NA values
map(data, function(.x) sum(is.na(.x)))

#check proportions of income groups
prop.table(table(data$class))

#plot proportion
options(repr.plot.width=4, repr.plot.height=4)
ggplot(data, aes(x=class))+geom_bar(fill="blue",alpha=0.5)+theme_bw()+labs(title="Distribution of Class")

#The most variables in the dataset are normally distributed as shown in the below plot, execpt for degree_spondylolisthesis:

data %>% plot_num(bins=10) 

#Check correlation

correlationMatrix <- cor(data[,0:6])
corrplot(correlationMatrix, order = "hclust", tl.cex = 1, addrect = 3)

#As seen in plot there seems to be no variables that are highly correlated with each others, cor >= 0.9
#due to this we can assume that methods that usually fail due to high correlation variable shouldn't be impacted on badly by the current variables
#The Caret R package provides the findCorrelation which will analyze a correlation matrix of your data’s attributes report on attributes that can be removed. 
#Because of much correlation some machine learning models could fail.
#Following method below proves the assumption of no highly correlated variables

# find Variables that are highly corrected (>0.90)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.7)
# print indexes of highly correlated attributes
print(highlyCorrelated)
#There are no variables

#Selecting the right features in our data can mean the difference between mediocre performance with long training times 
#and great performance with short training times.

# Remove correlated variables
data2 <- data %>% select(!highlyCorrelated)
# number of columns after removing correlated variables
ncol(data2)

#Lost 1 value

#MODELING Approach

#PCA
PcaData <- prcomp(data[,0:6], center = TRUE, scale = TRUE)
plot(PcaData, type="l")

#summary
summary(PcaData)

# only need the first component to explain 0.541 of the variance, after applying 4 principle components 0.94566 of the variance can be explained. According to the 
# summary above 1 of the variance can be explained after 5 principle components which 

#plot of PC1 vs PC2
pcaDf <- as.data.frame(PcaData$x)
ggplot(pcaDf, aes(x=PC1, y=PC2, col=data$class)) + geom_point(alpha=0.5)
#Kind of easy to separate into two classes


#plot of densitys
pc1 <- ggplot(pcaDf, aes(x=PC1, fill=data$class)) + geom_density(alpha=0.25)  
pc2 <- ggplot(pcaDf, aes(x=PC2, fill=data$class)) + geom_density(alpha=0.25)  

grid.arrange(pc1, pc2, ncol=2)

#NEW

PCAData2 <- prcomp(data2[,0:5], center = TRUE, scale = TRUE)
plot(PCAData2, type="l")


summary(PCAData2)

#The above table shows that 95% of the variance is explained with 4 PC's in the transformed dataset data2.

PcaDf2 <- as.data.frame(PCAData2$x)
ggplot(PcaDf2, aes(x=PC1, y=PC2, col=data$class)) + geom_point(alpha=0.5)

#The data of the first 2 components can be easly separated into two classes. This is caused by the fact that the variance explained by these components is not large. The data can be easly separated.

pc12 <- ggplot(PcaDf2, aes(x=PC1, fill=data$class)) + geom_density(alpha=0.25)  
pc22 <- ggplot(PcaDf2, aes(x=PC2, fill=data$class)) + geom_density(alpha=0.25)  
grid.arrange(pc12, pc22, ncol=2)

#Linear Discriminant Analysis (LDA)

#OG


LdaData <- MASS::lda(class~., data = data, center = TRUE, scale = TRUE) 
LdaData
#Data frame of the LDA for visualization purposes
ldaDataPredict <- predict(LdaData, data)$x %>% as.data.frame() %>% cbind(class=data$class)

#density plot
ggplot(ldaDataPredict, aes(x=LD1, fill=class)) + geom_density(alpha=0.5)

#Clean


LdaData2 <- MASS::lda(class~., data = data2, center = TRUE, scale = TRUE) 
LdaData2
#Data frame of the LDA for visualization purposes
ldaDataPredict2 <- predict(LdaData2, data2)$x %>% as.data.frame() %>% cbind(class=data2$class)

ggplot(ldaDataPredict2, aes(x=LD1, fill=class)) + geom_density(alpha=0.5)

# Methods

#creating train and training/validation datasets

#OG

set.seed(1, sample.kind="Rounding") #if using R 3.5 or earlier, use `set.seed(1)
#Train (80% of data)
data3 <- cbind (class=data$class, data2)
datSamplingIndex <- createDataPartition(data$class, times=1, p=0.8, list = FALSE)
trainData <- data3[datSamplingIndex, ]

#Test (20%)

testData <- data3[-datSamplingIndex, ]

#Control the data

fitControl <- trainControl(method="cv",    #Control the computational nuances of thetrainfunction
                           number = 20,    #Either the number of folds or number of resampling iterations
                           classProbs = TRUE,
                           summaryFunction = twoClassSummary)

#clean
#Train (80% of data)
data3 <- cbind (class=data$class, data2)
trainData2 <- data3[datSamplingIndex, ]
#Test (20%)
testData2 <- data3[-datSamplingIndex, ]


fitControl2 <- trainControl(method="cv",    
                            number = 20,    
                            classProbs = TRUE,
                            summaryFunction = twoClassSummary)

# Naive Bayes Model

#OG
NaiveBayesModel <- train(class~.,
                      trainData,
                      method="nb",
                      metric="ROC",
                      preProcess=c('center', 'scale'), #in order to normalize the data
                      trace=FALSE, 
                      importance = TRUE,
                      trControl=fitControl)

NBPrediction <- predict(NaiveBayesModel, testData)
NBConfMatrix <- confusionMatrix(NBPrediction, as.factor(testData$class), positive = "Abnormal")
NBConfMatrix


#The most important variables that permit the best prediction and contribute the most to the model are the following:

#ERORRRRRRRRRRRRRRRR

###plot(varImp(NaiveBayesModel)) Doesn't work


#clean 

NaiveBayesModel2 <- train(class~.,
                          trainData2,
                          method="nb",
                          metric="ROC",
                          preProcess=c('center', 'scale'), #in order to normalize the data
                          trace=FALSE, 
                          importance = TRUE,
                          trControl=fitControl2)

NBPrediction2 <- predict(NaiveBayesModel2, testData2)
NBConfMatrix2 <- confusionMatrix(NBPrediction2, as.factor(testData2$class), positive = "Abnormal")
NBConfMatrix2

# Random Forest

#OG
RandomforestModel <- train(class~.,
                            trainData,
                            method="rf",  #also recommended ranger, because it is a lot faster than original randomForest (rf)
                            metric="ROC",
                            #tuneLength=10,
                            #tuneGrid = expand.grid(mtry = c(2, 3, 6)),
                            preProcess = c('center', 'scale'),
                            trControl=fitControl)

RFPrediction <- predict(RandomforestModel, testData)
#Check results
RFConfMatrix <- confusionMatrix(RFPrediction, as.factor(testData$class), positive = "Abnormal")
RFConfMatrix

#plot importance
varImp(RandomforestModel)

plot(varImp(RandomforestModel), main="Ranking Of Importance")

#clean

RandomforestModel2 <- train(class~.,
                            trainData2,
                            method="rf",
                            metric="ROC",
                            preProcess = c('center', 'scale'),
                            trControl=fitControl2)

RFPrediction2 <- predict(RandomforestModel2, testData2)
#Check results
RFConfMatrix2 <- confusionMatrix(RFPrediction2, as.factor(testData2$class), positive = "Abnormal")
RFConfMatrix2

varImp(RandomforestModel2)

plot(varImp(RandomforestModel2), main="Ranking Of Importance")


# fit12_rf <- train(class ~., 
#                   data = trainData,
#                   method = "rf", 
#                   tuneGrid = data.frame(mtry = seq(1, 7)), 
#                   ntree = 100)
# fit12_rf$bestTune
# survived_hat <- predict(fit12_rf, testData)
# mean(survived_hat == testData$class)
# varImp(fit12_rf)


# Logistic Regression Model 

#OG 
LogRegModel<- train(class~., data = trainData, 
                     method = "glm",
                     metric = "ROC",
                     preProcess = c("scale", "center"),  # in order to normalize the data
                     trControl= fitControl)

LogRegPred <- predict(LogRegModel, testData)
# Check results
LogRegConfMatrix <- confusionMatrix(LogRegPred, as.factor(testData$class), positive = "Abnormal")
LogRegConfMatrix

#plot of importance
varImp(LogRegModel)
plot(varImp(LogRegModel), main="Ranking Of Importance")

#Clean
LogRegModel2<- train(class~., data = trainData2, 
                     method = "glm",
                     metric = "ROC",
                     preProcess = c("scale", "center"),  # in order to normalize the data
                     trControl= fitControl2)

LogRegPred2 <- predict(LogRegModel2, testData2)
# Check results
LogRegConfMatrix2 <- confusionMatrix(LogRegPred2, as.factor(testData2$class), positive = "Abnormal")
LogRegConfMatrix2

#var Importance
varImp(LogRegModel2)

plot(varImp(LogRegModel2), main="Ranking Of Importance")



# K Nearest Neighbor (KNN) Model
#OG
KNNModel <- train(class~.,
                   trainData,
                   method="knn",
                   metric="ROC",
                   preProcess = c('center', 'scale'),
                   tuneLength=10, #The tuneLength parameter tells the algorithm to try different default values for the main parameter
                   #In this case we used 10 default values
                   trControl=fitControl)

KNNPred <- predict(KNNModel, testData)
KNNConfMatrix <- confusionMatrix(KNNPred, as.factor(testData$class), positive = "Abnormal")
KNNConfMatrix

#plot importance

#plot(varImp(KNNModel),main="Top variables - KNN") Doesn't work

#Plot the kNN model to investigate the relationship between the number of neighbors and accuracy on the training set.

ggplot(KNNModel)

#RMSE(KNNPred,testData$class) don't use

#clean

KNNModel2 <- train(class~.,
                   trainData2,
                   method="knn",
                   metric="ROC",
                   preProcess = c('center', 'scale'),
                   tuneLength=10, 
                   trControl=fitControl2)

KNNPred2 <- predict(KNNModel2, testData2)
KNNConfMatrix2 <- confusionMatrix(KNNPred2, as.factor(testData2$class), positive = "Abnormal")
KNNConfMatrix2

print(ggplot(KNNModel2))

# TrainDatas <- data[,1:6]
# TrainClasses <- data[,6]
# 
# knnFit <- train(TrainDatas, TrainClasses, "knn")
# 
# knnImp <- varImp(knnFit)

#dotPlot(knnImp)


# Neural Network with LDA Model

#OG
#test and training data
trainDataLda <- ldaDataPredict[datSamplingIndex, ]
testDataLda <- ldaDataPredict[-datSamplingIndex, ]

# Model
NNLDAModel <- train(class~., 
                 trainDataLda,
                 method="nnet",
                 metric="ROC",
                 preProcess=c('center', 'scale'),
                 tuneLength=10,
                 trace=FALSE,
                 trControl=fitControl)

NNLdaPred <- predict(NNLDAModel, testDataLda)
NNLdaConfMatrix <- confusionMatrix(NNLdaPred, as.factor(testDataLda$class), positive = "Abnormal")
NNLdaConfMatrix

#Clean

trainDataLda2 <- ldaDataPredict[datSamplingIndex, ]
testDataLda2 <- ldaDataPredict[-datSamplingIndex, ]

#model

NNLDAModel2 <- train(class~., 
                     trainDataLda2,
                     method="nnet",
                     metric="ROC",
                     preProcess=c('center', 'scale'),
                     tuneLength=10,
                     trace=FALSE,
                     trControl=fitControl2)

NNLdaPred2 <- predict(NNLDAModel2, testDataLda2)
NNLdaConfMatrix2 <- confusionMatrix(NNLdaPred2, as.factor(testDataLda2$class), positive = "Abnormal")
NNLdaConfMatrix2


# Neural Network with PCA Model
#OG
NNPCAModel <- train(class~.,
                    trainData,
                    method="nnet",
                    metric="ROC",
                    preProcess=c('center', 'scale', 'pca'),
                    tuneLength=10,
                    trace=FALSE,
                    trControl=fitControl)

NNPcaPred <- predict(NNPCAModel, testData)
NNPcaConfMatrix <- confusionMatrix(NNPcaPred, as.factor(testData$class), positive = "Abnormal")
NNPcaConfMatrix


varImp(NNPCAModel)

plot(varImp(NNPCAModel), main="Ranking Of Importance")

#Clean
NNPCAModel2 <- train(class~.,
                     trainData2,
                     method="nnet",
                     metric="ROC",
                     preProcess=c('center', 'scale', 'pca'),
                     tuneLength=10,
                     trace=FALSE,
                     trControl=fitControl2)

NNPcaPred2 <- predict(NNPCAModel2, testData2)
NNPcaConfMatrix2 <- confusionMatrix(NNPcaPred2, as.factor(testData2$class), positive = "Abnormal")
NNPcaConfMatrix2

#var importance
varImp(NNPCAModel2)

plot(varImp(NNPCAModel2), main="Ranking Of Importance")

#Results
# comparisons

#OG
models <- list(NaiveBayes = NaiveBayesModel, 
               LogReg = LogRegModel,
               RandomForest = RandomforestModel,
               KNN = KNNModel,
               NeuralPCA = NNPCAModel,
               NeuralLDA = NNLDAModel)   

modelsResults <- resamples(models)
summary(modelsResults)

print(bwplot(modelsResults, metric="ROC",main = "Variablities"))

confMatrixs <- list(
  NaiveBayes = NBConfMatrix, 
  LogReg = LogRegConfMatrix,
  RandomForest = RFConfMatrix,
  KNN = KNNConfMatrix,
  NeuralPCA = NNPcaConfMatrix,
  NeuralLDA = NNPcaConfMatrix)   

ConfMatrixResults <- sapply(confMatrixs, function(x) x$byClass)
ConfMatrixResults %>% knitr::kable()

#Clean
models2 <- list(NaiveBayes = NaiveBayesModel2, 
                LogReg = LogRegModel2,
                RandomForest = RandomforestModel2,
                KNN = KNNModel2,
                NeuralPCA = NNPCAModel2,
                NeuralLDA = NNLDAModel2)   

modelsResults2 <- resamples(models2)
summary(modelsResults2)

#Plot variablities
bwplot(modelsResults2, metric="ROC",main = "Variablities")

confMatrixs2 <- list(
  NaiveBayes = NBConfMatrix2, 
  LogReg = LogRegConfMatrix2,
  RandomForest = RFConfMatrix2,
  KNN = KNNConfMatrix2,
  NeuralPCA = NNPcaConfMatrix2,
  NeuralLDA = NNPcaConfMatrix2)   

ConfMatrixResults2 <- sapply(confMatrixs2, function(x) x$byClass)
ConfMatrixResults2 %>% knitr::kable()




confMatrixs <- list(
  Naive_Bayes = NBConfMatrix, 
  Logistic_regr = LogRegConfMatrix,
  Random_Forest = RFConfMatrix,
  KNN = KNNConfMatrix,
  Neural_PCA = NNPcaConfMatrix,
  Neural_LDA = NNPcaConfMatrix)   

ConfMatrixResults <- sapply(confMatrixs, function(x) x$byClass)
ConfMatrixResults %>% knitr::kable()


#discussion 

#original
MaxConfMatrix <- apply(ConfMatrixResults, 1, which.is.max)
OutputReport <- data.frame(metric=names(MaxConfMatrix), 
                           bestModel=colnames(ConfMatrixResults)[MaxConfMatrix],
                           value=mapply(function(x,y) {ConfMatrixResults[x,y]}, 
                                        names(MaxConfMatrix), 
                                        MaxConfMatrix))
rownames(OutputReport) <- NULL
OutputReport

#clean
MaxConfMatrix2 <- apply(ConfMatrixResults2, 1, which.is.max)
OutputReport2 <- data.frame(metric=names(MaxConfMatrix2), 
                            bestModel=colnames(ConfMatrixResults2)[MaxConfMatrix2],
                            value=mapply(function(x,y) {ConfMatrixResults2[x,y]}, 
                                         names(MaxConfMatrix2), 
                                         MaxConfMatrix2))
rownames(OutputReport2) <- NULL
OutputReport2



# Appendix - Environment

print("Operating System:")
version

