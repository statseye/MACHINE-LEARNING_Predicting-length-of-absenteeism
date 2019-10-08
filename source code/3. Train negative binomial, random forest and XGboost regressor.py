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
