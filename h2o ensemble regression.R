#train<-read.csv('train.csv')
#test<-read.csv('test.csv')

index=sample(x = nrow(pp_train),size = 5000)

X_train=pp_train[index,]
y_test=pp_train[-index,]


train<-X_train
test<-y_test

index<-sample(1:(dim(train)[1]), 0.2*dim(train)[1], replace=FALSE)

train_frame<-train[-index,] 
valid_frame<-train[index,]  

valid_predict<-valid_frame[,-c(which(names(valid_frame) %in% c('SALES_PRICE')))]
#valid_loss<-valid_frame[,ncol(valid_frame)]  #y_test

# log transform
#train_frame[,ncol(train_frame)]<-log(train_frame[,ncol(train_frame)])
#valid_frame[,ncol(train_frame)]<-log(valid_frame[,ncol(valid_frame)])

# load H2o data frame // validate that H2O flow looses all continous data
#h2o.removeAll()
#h2o.init()

train_frame.hex<-as.h2o(train_frame)
valid_frame.hex<-as.h2o(valid_frame)
valid_predict.hex<-as.h2o(valid_predict)
test.hex<-as.h2o(y_test)
pp_test.hex=as.h2o(pp_test)
#----------------------------------------------------------
learner <- c( "h2o.randomForest.wrapper", 
             "h2o.gbm.wrapper", "h2o.glm.wrapper","h2o.deeplearning.wrapper")
learner <- c( "h2o.randomForest.wrapper",
              "h2o.gbm.wrapper","h2o.deeplearning.wrapper")

metalearner <-"h2o.glm.3"
metalearner <- "h2o.deeplearning.wrapper"

h2o.glm.1 <- function(..., alpha = 0.0) h2o.glm.wrapper(..., alpha = alpha)
h2o.glm.2 <- function(..., alpha = 0.5) h2o.glm.wrapper(..., alpha = alpha)
h2o.glm.3 <- function(..., alpha = 1.0) h2o.glm.wrapper(..., alpha = alpha)

learner <- c( "h2o.randomForest.wrapper", 
              "h2o.gbm.wrapper","h2o.deeplearning.wrapper","h2o.deeplearning.1", "h2o.deeplearning.6", "h2o.deeplearning.7")


fit <- h2o.ensemble(x = names(train_frame.hex)[-15], y = 'SALES_PRICE', training_frame  = 
                    train_frame.hex,model_id = "fit_ensemble",
                    family = "gamma", 
                    learner = learner, 
                    metalearner = metalearner,
                    cvControl = list(V = 10))	     



pred <- predict(fit,test.hex)
RMSE(as.integer(as.matrix(pred$pred['predict'])),y_test$SALES_PRICE)



# show stacked prediction and all 4 independent learners
# h2o.glm.wrapper h2o.randomForest.wrapper h2o.gbm.wrapper h2o.deeplearning.wrapper

# show combined only
head(pred[1])

class(as.integer(as.matrix(pred$pred['predict'])))
length(pred$pred)
RMSE(as.integer(as.matrix(pred$pred['predict'])),y_test$SALES_PRICE)
# show h2o.glm.wrapper
head(pred$basepred[1]) 



pp_test.hex=as.h2o(x = pp_test)
pred <- predict(fit,pp_test.hex)

submitlasso <- data.frame(PRT_ID=pp_temp1$PRT_ID,SALES_PRICE=as.integer(as.matrix(pred$pred['predict'])))
write.csv(submitlasso, "sub_saurabh2.csv",row.names = FALSE)

length(pp_test$AREA)
length(pp_temp1$PRT_ID)
