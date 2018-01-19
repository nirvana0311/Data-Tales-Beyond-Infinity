# Number of CV folds (to generate level-one data for stacking)
nfolds <- 5



# There are a few ways to assemble a list of models to stack toegether:
# 1. Train individual models and put them in a list
# 2. Train a grid of models
# 3. Train several grids of models
# Note: All base models must have the same cross-validation folds and
# the cross-validated predicted values must be kept.



# 1. Generate a 2-model ensemble (GBM + RF)

# Train & Cross-validate a GBM
my_gbm <- h2o.gbm(x = names(train_frame.hex)[-15], 
                  y = 'SALES_PRICE', 
                  training_frame  = train_frame.hex,
                  distribution = "gamma",
                  ntrees = 1000,
                  max_depth = 15,
                  min_rows = 20,
                  learn_rate = 0.05,
                  nfolds = nfolds,
                  fold_assignment = "Modulo",
                  keep_cross_validation_predictions = TRUE,
                  seed = 1)

pred <- predict(my_gbm,test.hex)
RMSE(as.integer(as.matrix(pred$pred['predict'])),y_test$SALES_PRICE)


# Train & Cross-validate a RF
my_rf <- h2o.randomForest(x = names(train_frame.hex)[-15], 
                          y = 'SALES_PRICE', 
                          training_frame  = train_frame.hex,
                          ntrees = 500,
                          nfolds = nfolds,
                          fold_assignment = "Modulo",
                          keep_cross_validation_predictions = TRUE,
                          seed = 1)
pred <- predict(my_rf,test.hex)
RMSE(as.integer(as.matrix(pred$pred['predict'])),y_test$SALES_PRICE)


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

#train_frame.hex<-as.h2o(train_frame)
train_frame.hex<-as.h2o(pp_train)
valid_frame.hex<-as.h2o(valid_frame)
valid_predict.hex<-as.h2o(valid_predict)
test.hex<-as.h2o(y_test)
pp_test.hex=as.h2o(pp_test)

mY_deepLearning <- h2o.deeplearning(x = names(train_frame.hex)[-15], 
                                    y = 'SALES_PRICE', 
                                    training_frame  = train_frame.hex,
                                    nfolds = 5,
                                    fold_assignment = "Modulo",
                                    model_id="dl_model_faster", 
                                    validation_frame=valid_frame.hex,
                                    hidden=c(15,25,35),                  ## small network, runs faster
                                    epochs=500,                      ## hopefully converges earlier...
                                    #score_validation_samples=10000,      ## sample the validation dataset (faster)
                                    #stopping_rounds=2,
                                    stopping_metric = 'RMSE', ## could be "MSE","logloss","r2"
                                    stopping_tolerance=0.01
                                  )

pred <- predict(mY_deepLearning,test.hex)
RMSE(as.integer(as.matrix(pred$pred['predict'])),y_test$SALES_PRICE)


#setwd("C:\\Users\\nirva\\IdeaProjects\\Analytics Vidhya")
#getwd()
pred <- predict(mY_deepLearning,pp_test.hex)
length(as.integer(as.matrix(pred$pred['predict'])))
submitlasso <- data.frame(PRT_ID=pp_temp1$PRT_ID,SALES_PRICE=as.integer(as.matrix(pred$pred['predict'])))
write.csv(submitlasso, "sub_last.csv",row.names = FALSE)






# Train a stacked ensemble using the GBM and RF above
ensemble <- h2o.stackedEnsemble(x = names(train_frame.hex)[-15], 
                                y = 'SALES_PRICE', 
                                training_frame  = train_frame.hex,
                                model_id = "my_ensemble_binomial",
                                base_models = list(my_gbm@model_id, my_rf@model_id,mY_deepLearning@model_id))

# Eval ensemble performance on a test set
perf <- h2o.performance(ensemble, newdata = test.hex)
pred <- predict(ensemble,test.hex)
RMSE(as.integer(as.matrix(pred$pred['predict'])),y_test$SALES_PRICE)



# Compare to base learner performance on the test set
perf_gbm_test <- h2o.performance(my_gbm, newdata = test)
perf_rf_test <- h2o.performance(my_rf, newdata = test)
baselearner_best_auc_test <- max(h2o.auc(perf_gbm_test), h2o.auc(perf_rf_test))
ensemble_auc_test <- h2o.auc(perf)
print(sprintf("Best Base-learner Test AUC:  %s", baselearner_best_auc_test))
print(sprintf("Ensemble Test AUC:  %s", ensemble_auc_test))

# Generate predictions on a test set (if neccessary)
pred <- h2o.predict(ensemble, newdata = test)