# 2. Generate a random grid of models and stack them together

# GBM Hyperparamters
learn_rate_opt <- c(0.01, 0.2,0.4)
max_depth_opt <- c(10,15)
sample_rate_opt <- c( 0.8)
col_sample_rate_opt <- c( 0.5,  0.8)
hyper_params <- list(learn_rate = learn_rate_opt,
                     max_depth = max_depth_opt,
                     sample_rate = sample_rate_opt,
                     col_sample_rate = col_sample_rate_opt)

search_criteria <- list(strategy = "RandomDiscrete",
                        max_models = 3,
                        seed = 1)

gbm_grid <- h2o.grid(algorithm = "gbm",
                     grid_id = "gbm_grid_binomial",
                     x = names(train_frame.hex)[-15], y = 'SALES_PRICE', training_frame  = 
                       train_frame.hex,
                     ntrees = 1000,
                     seed = 1,
                     nfolds = 5,
                     fold_assignment = "Modulo",
                     keep_cross_validation_predictions = TRUE,
                     hyper_params = hyper_params,
                     search_criteria = search_criteria)

# Train a stacked ensemble using the GBM grid
ensemble <- h2o.stackedEnsemble(x = names(train_frame.hex)[-15], 
                                y = 'SALES_PRICE', 
                                training_frame  = train_frame.hex,
                                model_id = "ensemble_gbm_grid_binomial",
                                base_models = gbm_grid@model_ids)


pred <- predict(fit,test.hex)
RMSE(as.integer(as.matrix(pred$pred['predict'])),y_test$SALES_PRICE)


# Eval ensemble performance on a test set
perf <- h2o.performance(ensemble, newdata = test.hex)

# Compare to base learner performance on the test set
.getauc <- function(mm) h2o.auc(h2o.performance(h2o.getModel(mm), newdata = test))
baselearner_aucs <- sapply(gbm_grid@model_ids, .getauc)
baselearner_best_auc_test <- max(baselearner_aucs)
ensemble_auc_test <- h2o.auc(perf)
print(sprintf("Best Base-learner Test AUC:  %s", baselearner_best_auc_test))
print(sprintf("Ensemble Test AUC:  %s", ensemble_auc_test))

# Generate predictions on a test set (if neccessary)
pred <- h2o.predict(ensemble, newdata = test)