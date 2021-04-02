library(tidyverse)
library(xgboost)
library(caret)
library(magrittr)
library(Matrix)
library(lightgbm)

getwd()
setwd ("/home/priyanka/work/pro3")

getwd()
set.seed(0)
R_earth <- 6371

tr <- read.csv("train_cab.csv")
tr <- tr[complete.cases(tr),]


tr  <- tr %>% 
  mutate(pickup_datetime = as.POSIXct(pickup_datetime)) %>%
  mutate(hour = as.numeric(format(pickup_datetime, "%H"))) %>%
  mutate(min = as.numeric(format(pickup_datetime, "%M"))) %>%   
  mutate(year = as.factor(format(pickup_datetime, "%Y"))) %>%
  mutate(day = as.factor(format(pickup_datetime, "%d"))) %>%
  mutate(month = as.factor(format(pickup_datetime, "%m"))) %>%
  mutate(Wday = as.factor(weekdays(pickup_datetime))) %>%
  mutate(hour_class = as.factor(ifelse(hour < 7, "Overnight", 
                                       ifelse(hour < 11, "Morning", 
                                              ifelse(hour < 16, "Noon", 
                                                     ifelse(hour < 20, "Evening",
                                                            ifelse(hour < 23, "night", "overnight") ) ))))) %>%
  filter(fare_amount > 0 & fare_amount <= 500) %>%
  filter(pickup_longitude > -80 && pickup_longitude < -70) %>%
  filter(pickup_latitude > 35 && pickup_latitude < 45) %>%
  filter(dropoff_longitude > -80 && dropoff_longitude < -70) %>%
  filter(dropoff_latitude > 35 && dropoff_latitude < 45) %>%
  filter(passenger_count > 0 && passenger_count < 10) %>%
  mutate(pickup_latitude = (pickup_latitude * pi)/180) %>%
  mutate(dropoff_latitude = (dropoff_latitude * pi)/180) %>%
  mutate(dropoff_longitude = (dropoff_longitude * pi)/180) %>%
  mutate(pickup_longitude = (pickup_longitude * pi)/180 ) %>%
  mutate(dropoff_longitude = ifelse(is.na(dropoff_longitude) == TRUE, 0,dropoff_longitude)) %>%
  mutate(pickup_longitude = ifelse(is.na(pickup_longitude) == TRUE, 0,pickup_longitude)) %>%
  mutate(pickup_latitude = ifelse(is.na(pickup_latitude) == TRUE, 0,pickup_latitude)) %>%
  mutate(dropoff_latitude = ifelse(is.na(dropoff_latitude) == TRUE, 0,dropoff_latitude)) %>%
  select(-pickup_datetime,-hour_class,-min)  

tr$dlat <- tr$dropoff_latitude - tr$pickup_latitude
tr$dlon <- tr$dropoff_longitude - tr$pickup_longitude 

#Compute haversine distance
tr$hav = sin(tr$dlat/2.0)**2 + cos(tr$pickup_latitude) * cos(tr$dropoff_latitude) * sin(tr$dlon/2.0)**2
tr$haversine <- 2 * R_earth * asin(sqrt(tr$hav))

#Compute Bearing distance
tr$dlon <- tr$pickup_longitude - tr$dropoff_longitude
tr$bearing <- atan2(sin(tr$dlon * cos(tr$dropoff_latitude)), cos(tr$pickup_latitude) * sin(tr$dropoff_latitude) - sin(tr$pickup_latitude) * cos(tr$dropoff_latitude) * cos(tr$dlon))


sphere_dist <- function(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)
{
  #Compute distances along lat, lon dimensions
  dlat = dropoff_lat - pickup_lat
  dlon = dropoff_lon - pickup_lon
  
  #Compute  distance
  a = sin(dlat/2.0)**2 + cos(pickup_lat) * cos(dropoff_lat) * sin(dlon/2.0)**2
  
  return (2 * R_earth * asin(sqrt(a)))
  
}   


# Latitude or longitude zero value mean fare
out <- tr %>%
  filter(pickup_longitude == 0 |  pickup_latitude == 0 | dropoff_latitude == 0 | dropoff_longitude ==0)

out_mean <- mean(out$fare_amount)


tr <- as.data.frame(tr) %>% 
  filter(!(pickup_longitude == 0 |  pickup_latitude == 0 | dropoff_latitude == 0 | dropoff_longitude ==0)) %>%
  select(-dlat,-dlon,-hav)

target <- tr$fare_amount
tr <- tr %>% select (- fare_amount)

cols <- colnames(tr)

categoricals.vec = colnames(tr)[c(grep("cat",colnames(tr)))]

tr$year <- as.numeric(tr$year)
tr$month <- as.numeric(tr$month)
tr$Wday <- as.numeric(tr$Wday)
tr$day <- as.numeric(tr$day)

tri <- createDataPartition(target, p = 0.9, list = F) %>% c()

dtrain <- Matrix(as.matrix(tr[tri, ]),sparse=TRUE)
dval <- Matrix(as.matrix(tr[-tri, ]),sparse=TRUE)



categorical_feature <- c("day","month","year")

lgb.train = lgb.Dataset(data=dtrain,label=target[tri],categorical_feature =categorical_feature)
lgb.valid = lgb.Dataset(data=dval,label=target[-tri],categorical_feature =categorical_feature)


lgb.grid = list(objective = "regression"
                , metric = "rmse"
                ,num_boost_round=10000
)

lgb.model <- lgb.train(
  params = lgb.grid
  , data = lgb.train
  , valids = list(val = lgb.valid)
  , learning_rate = 0.034
  , num_leaves = 31
  , max_depth = -1
  , subsample = .8
  , subsample_freq =1
  , colsample_bytree = 0.6
  , min_split_gain = 0.5
  , min_child_weight = 1
  , min_child_samples =10
  , scale_pos_weight = 1
  , num_threads = 4
  , boosting_type = "gbdt"
  , zero_as_missing = T
  , seed = 0
  , nrounds = 40000
  , early_stopping_rounds = 500
  , eval_freq = 50
)        

rm(tr, target, tri,dtrain,dval)
gc()


te <- read.csv("test.csv")

te  <- te %>% 
  mutate(pickup_datetime = as.POSIXct(pickup_datetime)) %>%
  mutate(hour = as.numeric(format(pickup_datetime, "%H"))) %>%
  mutate(min = as.numeric(format(pickup_datetime, "%M"))) %>%
  mutate(year = as.factor(format(pickup_datetime, "%Y"))) %>% 
  mutate(day = as.factor(format(pickup_datetime, "%d"))) %>%
  mutate(month = as.factor(format(pickup_datetime, "%m"))) %>%
  mutate(Wday = as.factor(weekdays(pickup_datetime))) %>%
  mutate(hour_class = ifelse(hour < 7, "Overnight", 
                             ifelse(hour < 11, "Morning", 
                                    ifelse(hour < 16, "Noon", 
                                           ifelse(hour < 20, "Evening",
                                                  ifelse(hour < 23, "night", "overnight") ) )))) %>%
  mutate(pickup_latitude = (pickup_latitude * pi)/180) %>%
  mutate(dropoff_latitude = (dropoff_latitude * pi)/180) %>%
  mutate(dropoff_longitude = (dropoff_longitude * pi)/180) %>%
  mutate(pickup_longitude = (pickup_longitude * pi)/180 ) %>%
  mutate(dropoff_longitude = ifelse(is.na(dropoff_longitude) == TRUE, 0,dropoff_longitude)) %>%
  mutate(pickup_longitude = ifelse(is.na(pickup_longitude) == TRUE, 0,pickup_longitude)) %>%
  mutate(pickup_latitude = ifelse(is.na(pickup_latitude) == TRUE, 0,pickup_latitude)) %>%
  mutate(dropoff_latitude = ifelse(is.na(dropoff_latitude) == TRUE, 0,dropoff_latitude)) %>%
  select(-pickup_datetime,-hour_class,-min)
te$dlat <- te$dropoff_latitude - te$pickup_latitude
te$dlon <- te$dropoff_longitude - te$pickup_longitude

#Compute haversine distance
te$hav = sin(te$dlat/2.0)**2 + cos(te$pickup_latitude) * cos(te$dropoff_latitude) * sin(te$dlon/2.0)**2
te$haversine <- 2 * R_earth * asin(sqrt(te$hav))


te$dlon <- te$pickup_longitude - te$dropoff_longitude


te$bearing = atan2(sin(te$dlon * cos(te$dropoff_latitude)), cos(te$pickup_latitude) * sin(te$dropoff_latitude) - sin(te$pickup_latitude) * cos(te$dropoff_latitude) * cos(te$dlon))    

te <- te %>% select(-dlat,-dlon,-hav)

te$year <- as.numeric(te$year)
te$month <- as.numeric(te$month)
te$Wday <- as.numeric(te$Wday)
te$day <- as.numeric(as.factor(te$day))

dtest1 <- Matrix(as.matrix(te),sparse=TRUE)

mutate(fare_amount = predict(lgb.model, dtest1)) %>%
write_csv("LightGBM_fare_amount.csv")
