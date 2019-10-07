### Run the Poisson model and check for overdispersion

import statsmodels.api as sm

# Add constant to set if independent variables
X_constant = sm.add_constant(X)

# Fit the Poisson model (the default link for the Poisson family is the log link)
poisson_reg = sm.GLM(y, X_constant, family = sm.families.Poisson()).fit()
print(poisson_reg.summary())

# The reported values of Deviance and Pearson chi-squared are large. 
# As per this test, the Poisson regression model fits the data rather poorly.

# Plot observed vs fitted values
from statsmodels.graphics.api import abline_plot

nobs = poisson_reg.nobs
yhat = poisson_reg.mu

fig, ax = plt.subplots()
ax.scatter(yhat, y)
ax.set_title('Model Fit Plot')
ax.set_ylabel('Observed values')
ax.set_xlabel('Fitted values');

### Run the Negative Binomial model

# The link functions currently implemented for the family of NegativeBinomial are the following:
sm.families.family.NegativeBinomial.links

# Fit NB model
neg_bin = sm.GLM(y, X_constant, family = sm.families.NegativeBinomial()).fit()
print(neg_bin.summary())
# Hence as per this test, the Negative Binomial regression does not fit the data well, 
# however, the fit seems to be better than with Poisson model.

# Plot observed vs fitted values
from statsmodels.graphics.api import abline_plot

nobs = neg_bin.nobs
yhat = neg_bin.mu

fig, ax = plt.subplots()
ax.scatter(yhat, y)
ax.set_title('Model Fit Plot')
ax.set_ylabel('Observed values')
ax.set_xlabel('Fitted values');

# Plotting the standardized deviance residuals to the predicted counts is another method of determining which model, Poisson or negative binomial, is a better fit for the data.

# Histograms of standardized deviance residuals

from scipy import stats

fig, ax = plt.subplots()
resid_Poisson = poisson_reg.resid_deviance.copy()
resid_Poisson_std = stats.zscore(resid_Poisson)
ax.hist(resid_Poisson_std, bins=25)
ax.set_title('Histogram of standardized deviance residuals (Poisson)');

fig, ax = plt.subplots()
resid_negbin = neg_bin.resid_deviance.copy()
resid_negbin_std = stats.zscore(resid_negbin)
ax.hist(resid_negbin_std, bins=25)
ax.set_title('Histogram of standardized deviance residuals (Negative Binomial)');

# The standardized deviance residuals to the predicted counts 

fig, ax = plt.subplots()
ax.scatter(poisson_reg.mu, resid_Poisson_std)
ax.set_title('Poisson model')
ax.set_ylabel('Standarized deviance residuals')
ax.set_xlabel('Fitted values');

fig, ax = plt.subplots()
ax.scatter(neg_bin.mu, resid_negbin_std)
ax.set_title('Negative Binomial model')
ax.set_ylabel('Standarized deviance residuals')
ax.set_xlabel('Fitted values');
# The model still has room for improvement.  That would require, if they are available, selecting better predictors of the outcome.
