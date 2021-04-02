library(dplyr)
library(mlbench)
library(mxnet)
setwd("G:/home/work/pro1")
Train <- read.csv(file = "train.csv", header = TRUE)
Test <- read.csv(file = "test.csv", header = TRUE)
Train$ID_code <- as.character(Train$ID_code)
Test$ID_code <- as.character(Test$ID_code)
Test$target <- as.integer(0)
Test <- Test[,c(1,202,2:201)]
VarList <- read.csv(file = "VarList0.csv", header = TRUE)
VarList$var <- as.character(VarList$var)
VarList$var <- gsub("var_", "", VarList$var)
VarList$var <- as.integer(VarList$var) + 3
VarList <- VarList[order(VarList$var),]
Train1 <- Train[,c(1,2)]
Test1 <- Test[,c(1,2)]
for (j in 1:length(VarList$var)) {
  D <- paste0("var_", VarList[j,1] - 3)
  Train1[,D] <- Train[,VarList[j,1]]
  Test1[,D] <- Test[,VarList[j,1]]
  j <- j + 1
}
Train <- Train1
Test <- Test1
for (i in 3:length(Train)) {
  Train[,i] <- as.integer(round(Train[,i]))
  Test[,i] <- as.integer(round(Test[,i]))
  i <- i + 1
}
cv <- Train
Train1 <- Train[Train$target == 1,]
for (i in 1:6) {
  
  Train1.y <- as.array(Train1[,2])
  Train1.x <- as.matrix(Train1[,3:42])
  cv.y <- as.array(cv[,2])
  cv.x <- as.matrix(cv[,3:42])
  Test.y <- as.array(cv[,2])
  Test.x <- as.matrix(cv[,3:42])
  
  model1 <- mx.mlp(Train1.x, Train1.y, hidden_node=mlConf[i,1], out_node=2, out_activation="softmax",
                   num.round=20, array.batch.size=1004, learning.rate=mlConf[i,2], momentum=0.9, 
                   eval.metric=mx.metric.accuracy)
  
  preds1 <- t(predict(model1, cv.x))
  preds1 <- as.data.frame(preds1)
  names(preds1) <- c(paste0("L1-",i, "0"),paste0("L1-",i, "1"))
  preds1[,1] <- as.integer(preds1[,1] * 100)
  preds1[,2] <- as.integer(preds1[,2] * 100)
  cv <- cbind(cv, preds1)
  
  preds1 <- t(predict(model1, Test.x))
  preds1 <- as.data.frame(preds1)
  names(preds1) <- c(paste0("L1-",i, "0"),paste0("L1-",i, "1"))
  preds1[,1] <- as.integer(preds1[,1] * 100)
  preds1[,2] <- as.integer(preds1[,2] * 100)
  Test <- cbind(Test, preds1)
  
  i <- i + 1
}

cv <- cv[,c(1,2,43:54)]
Test <- Test[,c(1,2,43:54)]

Train1 <- cv[cv$target == 1,]

for (i in 1:2) {
  
  Train1.y <- as.array(Train1[,2])
  Train1.x <- as.matrix(Train1[,3:14])
  cv.y <- as.array(cv[,2])
  cv.x <- as.matrix(cv[,3:14])
  Test.y <- as.array(cv[,2])
  Test.x <- as.matrix(cv[,3:14])
  
  model2 <- mx.mlp(Train1.x, Train1.y, hidden_node=128, out_node=2, out_activation="softmax",
                   num.round=20, array.batch.size=1004, learning.rate=0.01, momentum=0.9, 
                   eval.metric=mx.metric.accuracy)
  
  
  preds2 <- t(predict(model2, cv.x))
  preds2 <- as.data.frame(preds2)
  names(preds2) <- c(paste0("L2-",i, "0"),paste0("L2-",i, "1"))
  preds2[,1] <- as.integer(preds2[,1] * 100)
  preds2[,2] <- as.integer(preds2[,2] * 100)
  cv <- cbind(cv, preds2)
  
  preds2 <- t(predict(model2, Test.x))
  preds2 <- as.data.frame(preds2)
  names(preds2) <- c(paste0("L2-",i, "0"),paste0("L2-",i, "1"))
  preds2[,1] <- as.integer(preds2[,1] * 100)
  preds2[,2] <- as.integer(preds2[,2] * 100)
  Test <- cbind(Test, preds2)
  
  i <- i + 1
}
cv <- cv[,c(1,2,15:18)]
Test <- Test[,c(1,2,15:18)]

Train1 <- cv[cv$target == 1,]

Train1.y <- as.array(Train1[,2])
Train1.x <- as.matrix(Train1[,3:6])
cv.y <- as.array(cv[,2])
cv.x <- as.matrix(cv[,3:6])
Test.y <- as.array(cv[,2])
Test.x <- as.matrix(cv[,3:6])

model3 <- mx.mlp(Train1.x, Train1.y, hidden_node=64, out_node=2, out_activation="softmax",
                 num.round=20, array.batch.size=1004, learning.rate=0.05, momentum=0.9, 
                 eval.metric=mx.metric.accuracy)


preds3 <- t(predict(model3, Test.x))
preds3 <- as.data.frame(preds3)
names(preds3) <- c(paste0("L3-",i, "0"),paste0("L3-",i, "1"))
preds3[,1] <- as.integer(preds3[,1] * 100)
preds3[,2] <- as.integer(preds3[,2] * 100)
Test <- cbind(Test, preds3)

Test <- Test[,c(1,2,7:8)]

for (i in 1:length(Test$ID_code)) {
  if(Test[i,4] > Test[i,3]){
    Test[i,2] <- 1
  } else{}
  i <- i + 1
}

Test <- Test[,c(1,2)]
write.csv(Test, file = "mySubmission.csv", row.names = FALSE)