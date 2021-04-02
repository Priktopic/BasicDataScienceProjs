rm(list = ls());
graphics.off() 
library(readr)
library(datasets)
library(ggplot2)
library(forecast)
library(curl)
library(rstudioapi)
library(corrplot)
library(tseries)
library(dplyr)
library(tidyverse)
library(caret)
library(pander)
library(lubridate)
library(scales)

getwd()
setwd ("/home/priyanka/work/pro2")
getwd()  
rawData <- read.csv("day.csv")

processedData<-rawData
#processedData$<-week(processedData$dteday)
processedData$dteday<- NULL
processedData<-processedData[,c(2:15)]



## season wise ridership
season_count <- rawData %>%
select(season,cnt) 
point <- format_format(big.mark = ",", scientific = FALSE)
ggplot(season_count, aes(season, cnt))+ geom_bar(stat = "identity", fill="coral1") + labs(title="Seasonal ridership")+
scale_y_continuous(labels = point)

## year wise casual and registered ridership
Year_count<- rawData%>% 
select(yr,casual,registered)
df <- Year_count%>%
group_by(yr) %>%
summarise(casual_ridership=sum(casual),registered_ridership = sum(registered))
df <-as.data.frame(df)
df$yr <- as.character(df$yr)

dfm <- melt(df[,c('yr','casual_ridership','registered_ridership')],id.vars = 1)
point <- format_format(big.mark = ",", scientific = FALSE)
ggplot(dfm,aes(x = yr,y = value)) + labs(title="Yearly casual and registered ridership ")+
  geom_bar(aes(fill = variable),stat = "identity",position = "dodge") + scale_y_continuous(labels = point)


month_count<- rawData %>% 
  select(mnth,workingday,cnt)
df <- month_count%>%
  group_by(mnth,workingday) %>%
  summarise(cnt = sum(cnt))
df$mnth <- as.factor(df$mnth)
df$workingday <- as.character(df$workingday)
point <- format_format(big.mark = ",", scientific = FALSE)
ggplot(df, aes(mnth,cnt)) + labs(title="Monthly ridership based on working and holiday")+
  geom_line(aes(color=workingday, group=workingday))+ scale_y_continuous(labels = point)


## Ridership based on weather
weather_count <- rawData %>%
  select(mnth,weathersit,cnt)
weather_df <- weather_count %>%
  group_by(mnth,weathersit) %>%
  summarise(cnt = sum(cnt))
weather_df$mnth <- as.factor(weather_df$mnth)
weather_df$weathersit <- as.character(weather_df$weathersit)
point <- format_format(big.mark = ",", scientific = FALSE)
ggplot(weather_df, aes(mnth,cnt)) + labs(title="Monthly ridership based on weather")+
  geom_line(aes(color=weathersit, group=weathersit))+scale_y_continuous(labels = point)


rawData %>%
  mutate(weathersit= factor(weathersit)) %>%
  ggplot(aes(y=registered , x=weathersit, fill=weathersit))+geom_boxplot(colour="black")+labs(title="Registered ridership")+ scale_fill_discrete(name="Weather type", labels=c("Clear", "Mist", "Light Snow", "Heavy Rain"))

rawData %>%
  mutate(weathersit= factor(weathersit)) %>%
  ggplot(aes(y=casual , x=weathersit, fill=weathersit))+geom_boxplot(colour="black")+labs(title="Casual ridership")+scale_fill_discrete(name="Weather type", labels=c("Clear", "Mist", "Light Snow", "Heavy Rain"))

  
correlation <- mutate_all(rawData, function(x) as.numeric(as.character(x)))
df<- cor(rawData[,3:16])
corrplot(cor(df), method = 'circle')


# Scatter plot, Visualize the linear relationship between the predictor and response
scatter.smooth(x=rawData$temp, y=rawData$cnt, main="cnt ~ temperature")

#BoxPlot – Check for outliers
par(mfrow=c(1, 2))  # divide graph area in 2 columns
boxplot(rawData$temp, main="temperature", sub=paste("Outlier rows: ", boxplot.stats(rawData$temp)$out))  
boxplot(rawData$cnt, main="ridership", sub=paste("Outlier rows: ", boxplot.stats(rawData$cnt)$out))
cor(rawData$cnt, rawData$temp)


# Build Linear Model only based on temperature
lmfit1<-lm(rawData$cnt ~rawData$temp)
panderOptions("digits")
pander(lmfit1, caption = "Linear Model: bike riders ~ temp")


R1=summary(lmfit1)$r.squared
cat("R-Squared = ", R1)

ggplot(rawData, aes(temp, cnt)) + 
  geom_point(color="firebrick") +
  ggtitle('Ridership vs. temperature') +
  theme(plot.title = element_text(size=19.5, face="bold"))+
  labs(x="temperature", y="ridership")+
  theme(axis.text.x=element_text(angle=90, vjust=.5)) +
  theme(panel.background = element_rect(fill = 'grey75'))+
  stat_smooth(method = "lm",  formula = y ~ x, col = "yellow")


train_df <- rawData[1:547, ]
test_df <- rawData[547:nrow(rawData), ]

# Create a linear regression model
lm_model <- lm(cnt ~ temp+ workingday + weathersit + atemp + hum + windspeed, data = train_df)
print(lm_model)

# Predicting on test set
pred1Train <- predict(lm_model, test_df)

# Visualizing the linear regression Plot
plot(rawData$instant, rawData$cnt, type = "l", col = "red", xlab = "Day", ylab = "Number of Bike Users", main = "linear regression Plot for Bike Users")
legend("topleft", c("Actual", "Estimated"), lty = c(1, 1), col = c("red", "blue"))
lines(test_df$instant, pred1Train, type = "l", col = "blue")
write.csv(pred1Train, file="linearregressionoutput.csv", row.names = FALSE)

library(randomForest)
train_df <- rawData[1:547, ]
test_df <- rawData[547:nrow(rawData), ]

# Create a Random Forest model
rf_model <- randomForest(cnt ~ temp+ workingday + weathersit + atemp + hum + windspeed, data = train_df, ntree = 10)
print(rf_model)

# Predicting on test set
pred2Train <- predict(rf_model, test_df, type = "class")

# Visualizing the Random Forest Plot
plot(rawData$instant, rawData$cnt, type = "l", col = "red", xlab = "Day", ylab = "Number of Bike Users", main = "Random Forest Plot for Bike Users")
legend("topleft", c("Actual", "Estimated"), lty = c(1, 1), col = c("red", "blue"))
lines(test_df$instant, pred2Train, type = "l", col = "blue")
write.csv(pred2Train, file="randomforestoutput.csv", row.names = FALSE)

