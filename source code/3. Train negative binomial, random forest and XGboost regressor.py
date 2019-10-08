### Test predictive properties of the two models

# Create training and test datasets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_constant, y, test_size=0.2, random_state = 123)
X_test.info()
