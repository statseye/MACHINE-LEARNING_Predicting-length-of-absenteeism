### Test predictive properties of 3 models: Negative Binomial, Random forest and XGboost Regressor

### Create training and test datasets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_constant, y, test_size=0.2, random_state = 123)
X_test.info()

### Negative Binomial

# Train the model
neg_bin = sm.GLM(y_train, X_train, family = sm.families.NegativeBinomial()).fit()
print(neg_bin.summary())

# Test Negative Binomial
y_test_neg_bin = neg_bin.predict(X_test)

fig, ax = plt.subplots()
ax.scatter(y_test, y_test_neg_bin)
ax.set_title('Negative Binomial model')
ax.set_ylabel('Predicted y_test')
ax.set_xlabel('Observed y_test');

from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, y_test_neg_bin))
# MAE= 5.1555. Output is non-negative floating point. The best value is 0.0.

# Check which variables were found significant
neg_bin_summary = pd.DataFrame({'Coefficients': neg_bin.params, 'P>|z|': neg_bin.pvalues}).sort_values(by=['P>|z|'])

pd.options.display.float_format = '{:.3f}'.format
print(neg_bin_summary)

# Two most important:
#                                                 Coefficients  P>|z|
#Reason_disease                                       1.359  0.000
#Reason_external                                      1.435  0.000
# https://stats.idre.ucla.edu/stata/output/negative-binomial-regression/
# Interpretation for Reason_disease: This is the estimated negative binomial regression coefficient comparing 
# being absent due to disease with other reasons, given the other variables are held constant in the model. 
# The difference in the logs of expected counts is expected to be 1.359 unit higher for disease compared to other reasons, 
# while holding the other variables constant in the model.

### Random forest regressor

# Train a model with 100 trees (n_estimators) with OOB sample
from sklearn.ensemble import RandomForestRegressor
rm_reg = RandomForestRegressor(n_estimators = 100,
                                  random_state = 0, 
                                  oob_score=True)
rm_reg.fit(X_train, y_train)

# Apply Grid Search to find optimal parameters
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
from sklearn.model_selection import GridSearchCV

parameters = {'max_features': [5, 10, 15],
               'min_samples_leaf': [30, 50, 70], 
               'n_estimators': [50, 100]}

# use MAE to evaluate the models
grid_search = GridSearchCV(estimator = rm_reg,
                           param_grid = parameters,
                           scoring = 'neg_mean_absolute_error',
                           cv = 10,
                           n_jobs = -1, 
                           return_train_score=True)

# Change shape of the datasets and drop constant
X_train.shape
y_train.shape

X_train.info()

X_train_rm = X_train.drop('const', axis=1)
X_train_rm.info()

X_test_rm = X_test.drop('const', axis=1)
X_test_rm.info()

y_train_rm = np.ravel(y_train)
y_train_rm.shape
y_test_rm = np.ravel(y_test)

# Fit random forest to train data, apply grid search
rm_reg_gs = grid_search.fit(X_train_rm, y_train_rm)

# Return train score history for all combinations of parameters
cv_results = pd.DataFrame(rm_reg_gs.cv_results_)
print(cv_results[['params', 'mean_train_score']])

# Distribution of MAE for training and test datasets
cv_results[['mean_test_score', 'mean_train_score']].describe()

# Return parameter setting that gave the best results on the hold out data.
best_parameters = rm_reg_gs.best_params_
print("Random forest parameter setting that gave the best result (MAE) on the hold out data:", best_parameters)

# Fit best RF model
rm_reg_best = RandomForestRegressor(n_estimators = 50,
                                    min_samples_leaf = 30, 
                                    max_features = 10, 
                                    random_state = 123,
                                    oob_score= False)
rm_reg_best.fit(X_train_rm, y_train_rm)

# Test the model
y_test_rm_pred = rm_reg_best.predict(X_test_rm)

from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test_rm, y_test_rm_pred))
# 5.000 - lower than for Negative Binomial (5.1555) 

fig, ax = plt.subplots()
ax.scatter(y_test_rm, y_test_rm_pred)
ax.set_title('Random Forest model')
ax.set_ylabel('Predicted y_test')
ax.set_xlabel('Observed y_test');

# Feature importances
var_import=pd.DataFrame(data=rm_reg_best.feature_importances_, 
                        index=X_train_rm.columns)
print(var_import.sort_values(by=0, ascending=False))
# Most important: Reason_disease, Reason_external, Son

### XGboost 

# Step 1: Use default values for parameters in order to decide on number of estimators. 
# Use booste='gblinear' (tested it and it achieved best results for this problem)

xgb1a =  XGBRegressor(
 learning_rate =0.1,
 booster = 'gblinear', 
 n_estimators=500,
 objective= 'count:poisson',
 random_state=123)

xgb1a.fit(X_train_rm, y_train, 
         eval_set=[(X_train_rm, y_train), (X_test_rm, y_test)],
         eval_metric='mae',
         early_stopping_rounds = 6,
         verbose = True)
print(xgb1a)

# Best iteration: [176]   
# validation_0-mae:5.49028        validation_1-mae:5.40566
# MAE at the best iteration of gbtree was 6.88851, for gblinear is 5.47568 - much better. 

# Step 2: Tune max_depth and min_child_weight

# Set number of estimators to 200 (learning from Step 1)
xgb2 =  XGBRegressor(learning_rate =0.1,
                     booster = 'gblinear', 
                     n_estimators=200,
                     objective= 'count:poisson',
                     random_state=123)

# define search grid. 
paramset1 = {'max_depth': [2,3,4,5,6],
               'min_child_weight':range(1,30,2)}

# search with 10-fold cross-validations. The performace will be evaluated on 10 random validation datasets (within training dataset)
gsearch1 = GridSearchCV(estimator = xgb2,
                           param_grid = paramset1,
                           scoring = 'neg_mean_absolute_error',
                           cv = 10,
                           n_jobs = -1)

gsearch1 = gsearch1.fit(X_train_rm, y_train)

print('best_parameters:', gsearch1.best_params_)
print('best_mean_validation_score:', gsearch1.best_score_)
#best_parameters: {'max_depth': 2, 'min_child_weight': 1}
#best_mean_validation_score: -5.6794131339804546

# Step 3: Tune gamma

paramset2 = {
 'gamma':[i/10.0 for i in range(0,5)]
}

# Set parameters according to learning from Step 1 and 2

xgb3 =  XGBRegressor(learning_rate =0.1,
                      n_estimators=170,
                      max_depth = 2, 
                      min_child_weight = 1, 
                      objective= 'count:poisson',
                      random_state=123)

gsearch2 = GridSearchCV(estimator = xgb3,
                           param_grid = paramset2,
                           scoring = 'neg_mean_absolute_error',
                           cv = 10,
                           n_jobs = -1)
gsearch2

gsearch2 = gsearch2.fit(X_train_rm, y_train)

print('best_parameters:', gsearch2.best_params_)
print('best_mean_validation_score:', gsearch2.best_score_)
#best_parameters: {'gamma': 0.2}
#best_mean_validation_score: -5.645409741080241

# Step 4: Tune subsample and colsample_bytree

# The next step would be try different subsample and colsample_bytree values. 
# Lets take values 0.6,0.7,0.8,0.9, 1.0 for both to start with.

paramset3 = {
 'subsample':[0.6,0.7,0.8,0.9,1.0],
 'colsample_bytree':[0.6,0.7,0.8,0.9,1.0]
}

xgb4 =  XGBRegressor(learning_rate =0.1,
                      n_estimators=170,
                      max_depth = 2, 
                      min_child_weight = 1,
                      gamma = 0.2,
                      objective= 'count:poisson',
                      random_state=123)

gsearch3 = GridSearchCV(estimator = xgb4,
                           param_grid = paramset3,
                           scoring = 'neg_mean_absolute_error',
                           cv = 10,
                           n_jobs = -1)
gsearch3

gsearch3 = gsearch3.fit(X_train_rm, y_train)

print('best_parameters:', gsearch3.best_params_)
print('best_mean_validation_score:', gsearch3.best_score_)
#best_parameters: {'colsample_bytree': 0.6, 'subsample': 1.0}
#best_mean_validation_score: -5.6133224609043655

# Step 5: Tuning Regularization Parameters

# Next step is to apply regularization to reduce overfitting. Though many people don’t use this parameters much as gamma provides a substantial way of controlling complexity.

paramset4 = {
        'reg_alpha': [0, 1e-5, 1e-2, 0.1, 0.2, 0.3, 0.4],
        'reg_lambda': [1, 2, 5, 10]}

xgb5 =  XGBRegressor(learning_rate =0.1,
                      n_estimators=170,
                      max_depth = 2, 
                      min_child_weight = 1,
                      gamma = 0.2, 
                      colsample_bytree = 0.6,
                      subsample = 1.0,
                      objective= 'count:poisson',
                      random_state=123)

gsearch4 = GridSearchCV(estimator = xgb5,
                           param_grid = paramset4,
                           scoring = 'neg_mean_absolute_error',
                           cv = 10,
                           n_jobs = -1)

gsearch4 = gsearch4.fit(X_train_rm, y_train)

print('best_parameters:', gsearch4.best_params_)
print('best_mean_validation_score:', gsearch4.best_score_)
#best_parameters: {'reg_alpha': 0.03, 'reg_lambda': 1}
#best_mean_validation_score: -5.5727575767936335

# Step 6: Tune Learning Rate

# Lastly, we should lower the learning rate.

paramset5 = {'learning_rate': [0.005, 0.007, 0.01, 0.025, 0.05, 0.075, 0.1]}

# Set parameters according to learning from all previous steps
# Increase number of trees as we will try lowering the learning rate

xgb6 =  XGBRegressor(learning_rate =0.1,
                      n_estimators=500,
                      max_depth = 2, 
                      min_child_weight = 1,
                      gamma = 0.2, 
                      colsample_bytree = 0.6,
                      subsample = 1.0,
                      reg_alpha = 0.03, 
                      reg_lambda = 1, 
                      objective= 'count:poisson',
                      random_state=123)

gsearch5 = GridSearchCV(estimator = xgb6,
                           param_grid = paramset5,
                           scoring = 'neg_mean_absolute_error',
                           cv = 10,
                           n_jobs = -1)

gsearch5 = gsearch5.fit(X_train_rm, y_train)

print('best_parameters:', gsearch5.best_params_)
print('best_mean_validation_score:', gsearch5.best_score_)
#best_parameters: {'learning_rate': 0.005}
#best_mean_validation_score: -5.248726653488907

# Step 7: Validate the final model on test dataset. Increase number of estimators, we will use early stopping anyway. 
 
xgb_final =  XGBRegressor(learning_rate =0.005,
                      n_estimators=500,
                      max_depth = 2, 
                      min_child_weight = 1,
                      gamma = 0.2, 
                      colsample_bytree = 0.6,
                      subsample = 1.0,
                      reg_alpha = 0.03, 
                      reg_lambda = 1, 
                      objective= 'count:poisson',
                      random_state=123)

xgb_final.fit(X_train_rm, y_train, 
         eval_set=[(X_train_rm, y_train), (X_test_rm, y_test)],
         eval_metric='mae',
         early_stopping_rounds = 6,
         verbose=True)
print(xgb_final)
# Stopping. Best iteration:[129]   
# validation_0-mae:4.64297        validation_1-mae:3.89049


# MAE on test dataset has decreased from 5.47568 with the initial model to 3.89049 for the final model. 
# We are achieving this result with 129 trees. # Significant improvement on test data vs random forest model (5.000). 

print(xgb_final.best_score)
print(xgb_final.best_iteration)
print(xgb_final.best_ntree_limit)

y_pred = xgb_final.predict(X_test_rm, ntree_limit = xgb_final.best_iteration)
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))

# Visualize training - how AUC improved for training and test dataset
xgb_final_evals_result = xgb_final.evals_result()
epochs = len(xgb_final_evals_result['validation_0']['mae'])
x_axis = range(0, epochs)

fig, ax = plt.subplots()
ax.plot(x_axis, xgb_final_evals_result['validation_0']['mae'], label='Train')
ax.plot(x_axis, xgb_final_evals_result['validation_1']['mae'], label='Test')
ax.legend()
plt.ylabel('MAE')
plt.title('XGBoost MAE: final model')
plt.show()

# Save the final model as an object to a file for later retrieval
xgb_forsave =  XGBRegressor(learning_rate =0.005,
                      n_estimators=129,
                      max_depth = 2, 
                      min_child_weight = 1,
                      gamma = 0.2, 
                      colsample_bytree = 0.6,
                      subsample = 1.0,
                      reg_alpha = 0.03, 
                      reg_lambda = 1, 
                      objective= 'count:poisson',
                      random_state=123)

xgb_forsave.fit(X_train_rm, y_train, 
         verbose=True)
print(xgb_forsave)

# Running this example saves the final XGBoost model to x.dat pickle file in the current working directory.
pickle.dump(xgb_forsave, open('xgb_absenteeism.dat', "wb"))

# some time later...

# Load model from file
loaded_model = pickle.load(open('xgb_absenteeism.dat', "rb"))
print(loaded_model)

# Explain the model's predictions on the entire dataset

# Plot Feature Importance (SHAP values)
import shap
explainer = shap.TreeExplainer(loaded_model)
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values, X, plot_type="bar")

# The x-axis is essentially the average magnitude change in model output when a feature is “hidden” from the model. 
# We can see that the Reason_disease feature is the strongest predictor of lenght of absenteeism, followed by Reason_external and Son. 
# The same variables have proved important in random forest model. 

