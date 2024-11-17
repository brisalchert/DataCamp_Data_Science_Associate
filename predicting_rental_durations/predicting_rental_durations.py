#-----------------------------------------------------------------------------------------------------------------------
#  In this project, you will use regression models to predict the number of days a customer rents DVDs for.
#
#  As with most data science projects, you will need to pre-process the data provided, in this case, a csv file called
#  rental_info.csv. Specifically, you need to:
#
#  - Read in the csv file rental_info.csv using pandas.
#  - Create a column named "rental_length_days" using the columns "return_date" and "rental_date", and add it to the
#    pandas DataFrame. This column should contain information on how many days a DVD has been rented by a customer.
#  - Create two columns of dummy variables from "special_features", which takes the value of 1 when:
#    - The value is "Deleted Scenes", storing as a column called "deleted_scenes".
#    - The value is "Behind the Scenes", storing as a column called "behind_the_scenes".
#  - Make a pandas DataFrame called X containing all the appropriate features you can use to run the regression models,
#    avoiding columns that leak data about the target.
#  - Choose the "rental_length_days" as the target column and save it as a pandas Series called y.
#
#  Following the preprocessing you will need to:
#
#  - Split the data into X_train, y_train, X_test, and y_test train and test sets, avoiding any features that leak data
#    about the target variable, and include 20% of the total data in the test set.
#  - Set random_state to 9 whenever you use a function/method involving randomness, for example, when doing a
#    test-train split.
#
#  Recommend a model yielding a mean squared error (MSE) less than 3 on the test set
#
#  - Save the model you would recommend as a variable named best_model, and save its MSE on the test set as best_mse.
#-----------------------------------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Lasso regression for feature selection
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

# OLS
from sklearn.linear_model import LinearRegression

# Random forest
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

# Load data
rentals = pd.read_csv("rental_info.csv")
print(rentals.head())

# Fix date data types
rentals["rental_date"] = pd.to_datetime(rentals["rental_date"])
rentals["return_date"] = pd.to_datetime(rentals["return_date"])

print(rentals.info())

# Add column for rental length
rentals["rental_length_days"] = (rentals["return_date"] - rentals["rental_date"]).dt.days

# Create dummy variables for special features
rentals["deleted_scenes"] = np.where(rentals["special_features"].str.contains("Deleted Scenes"), 1, 0)
rentals["behind_the_scenes"] = np.where(rentals["special_features"].str.contains("Behind the Scenes"), 1, 0)

# Create DataFrame of features for regression model
X = rentals[[
    "amount",
    "release_year",
    "rental_rate",
    "length",
    "replacement_cost",
    "NC-17",
    "PG",
    "PG-13",
    "R",
    "deleted_scenes",
    "behind_the_scenes"
]]

# Create series of targets for regression model
y = rentals["rental_length_days"]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=9)

# Scale training data for lasso model
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)

# Fit lasso model for feature selection
lasso = Lasso(alpha=0.3, random_state=9)
lasso.fit(X_train_scaled, y_train)
lasso_coefficients = lasso.coef_

# Select features with positive coefficients for model training
X_train, X_test = X_train.iloc[:, lasso_coefficients > 0], X_test.iloc[:, lasso_coefficients > 0]

# Fit linear regression model and get MSE
linreg = LinearRegression()
linreg.fit(X_train, y_train)
y_pred = linreg.predict(X_test)
linreg_mse = mean_squared_error(y_test, y_pred)
print(linreg_mse)

# Create hyperparameter grid for tuning
params_rf = {
    "n_estimators": np.arange(1, 101, 1),
    "max_depth": np.arange(1, 11, 1),
}

# Initialize random forest model
rf = RandomForestRegressor()

# Initialize randomized CV search for hyperparameters
random_search_rf = RandomizedSearchCV(rf, param_distributions=params_rf, cv=10, random_state=9)

# Fit the model and get the best model and parameters
random_search_rf.fit(X_train, y_train)
best_model = random_search_rf.best_estimator_
best_params = random_search_rf.best_params_

# Evaluate best model and get MSE
y_pred = best_model.predict(X_test)
rf_mse = mean_squared_error(y_test, y_pred)

print(best_params)
print(rf_mse)

# Save best model and lowest MSE
best_mse = min(linreg_mse, rf_mse)

if linreg_mse < rf_mse:
    best_model = linreg_mse
