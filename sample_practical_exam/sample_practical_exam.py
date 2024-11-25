# Use this cell to write your code for Task 1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

# Load dataset
loyalty = pd.read_csv("loyalty.csv")

# Examine missing values
msno.matrix(loyalty)
plt.show()

# Clean first_month column
loyalty.first_month.value_counts()
null_first_month = loyalty["first_month"] == "."
# Replace missing values with 0
loyalty["first_month"] = np.where(loyalty["first_month"] == ".", 0, loyalty["first_month"])
loyalty["first_month"] = loyalty["first_month"].astype("float")

# Clean region column
loyalty["region"] = loyalty["region"].astype("category")

# Clean loyalty_years column
categories = ["0-1", "1-3", "3-5", "5-10", "10+"]
loyalty["loyalty_years"] = loyalty["loyalty_years"].astype(pd.CategoricalDtype(categories, ordered=True))

# Clean joining_month column
null_joining = loyalty["joining_month"].isna()
# Replace missing values with "Unknown"
loyalty["joining_month"] = np.where(loyalty["joining_month"].isna(), "Unknown", loyalty["joining_month"])
loyalty["joining_month"] = loyalty["joining_month"].astype("category")

# Clean promotion column
loyalty["promotion"] = loyalty["promotion"].str.title().astype("category")

clean_data = loyalty
print(clean_data)



# Use this cell to write your code for Task 2
loyal = pd.read_csv("loyalty.csv")

# Group by years in programme and calculate avg spend and variance
spend_by_years = loyal.groupby("loyalty_years")["spend"].agg(["mean", "var"])
spend_by_years.rename(columns={"mean": "avg_spend", "var": "var_spend"}, inplace=True)
spend_by_years.reset_index(inplace=True)
spend_by_years["avg_spend"] = spend_by_years["avg_spend"].round(2)
spend_by_years["var_spend"] = spend_by_years["var_spend"].round(2)
print(spend_by_years)



# Use this cell to write your code for Task 3
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Isolate spend for both sets
y_train = train["spend"]
X_train = train.drop("spend", axis=1)

X_test = test

# One-hot encode categorical features
X_train = pd.get_dummies(X_train, drop_first=True)

X_test = pd.get_dummies(X_test, drop_first=True)

# Instantiate model
model = LinearRegression()

model.fit(X_train, y_train)

# Get RMSE for model
y_pred_train = model.predict(X_train)
rmse = MSE(y_train, y_pred_train)**(1/2)
print("RMSE train: ", rmse)

cv = -cross_val_score(model, X_train, y_train, cv=5, scoring="neg_mean_squared_error")
cv = np.sqrt(cv)
cv_rmse = np.mean(cv)
print("CV RMSE: ", cv_rmse)

# Predict spend for testing set
y_pred = model.predict(X_test)

base_result = pd.DataFrame({"customer_id": test["customer_id"], "spend": y_pred}).round(2)
print(base_result)



# Use this cell to write your code for Task 3
from sklearn.ensemble import GradientBoostingRegressor

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Isolate spend for both sets
y_train = train["spend"]
X_train = train.drop("spend", axis=1)

X_test = test

X_test = pd.get_dummies(X_test, drop_first=True)

# One-hot encode categorical features
X_train = pd.get_dummies(X_train, drop_first=True)

X_test = pd.get_dummies(X_test, drop_first=True)

# Instantiate model
model = GradientBoostingRegressor(n_estimators=200, max_depth=6, min_samples_leaf=10, random_state=42)

model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
rmse_grid = MSE(y_train, y_pred_train)**(1/2)
print("RMSE train: ", rmse_grid)

cv = -cross_val_score(model, X_train, y_train, cv=10, scoring="neg_mean_squared_error")
cv = np.sqrt(cv)
cv_rmse = np.mean(cv)
print("CV RMSE: ", cv_rmse)

# Predict spend for testing set
y_pred = model.predict(X_test)

compare_result = pd.DataFrame({"customer_id": test["customer_id"], "spend": y_pred}).round(2)
print(compare_result)
