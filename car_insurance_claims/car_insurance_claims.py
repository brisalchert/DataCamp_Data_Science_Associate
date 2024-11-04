#-----------------------------------------------------------------------------------------------------------------------
#  - Identify the single feature of the data that is the best predictor of whether a customer will put in a claim (the
#    "outcome" column), excluding the "id" column.
#
#  - Store as a DataFrame called best_feature_df, containing columns named "best_feature" and "best_accuracy" with the
#    name of the feature with the highest accuracy, and the respective accuracy score.
#-----------------------------------------------------------------------------------------------------------------------

# Import required modules
import pandas as pd
import numpy as np
from statsmodels.formula.api import logit
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns

# Load data from car_insurance.csv
insurance_df = pd.read_csv("car_insurance.csv")

# Explore data
print(insurance_df.head())
print(insurance_df.describe())
print(insurance_df.info())

# Check for missing values
print(insurance_df.isna().sum())
msno.matrix(insurance_df)
plt.show()

# Check distribution of columns with missing values
fig, ax = plt.subplots(2, 1)
sns.histplot(ax=ax[0], data=insurance_df, x="credit_score")
sns.histplot(ax=ax[1], data=insurance_df, x="annual_mileage")
plt.subplots_adjust(hspace=0.4, wspace=0.4)
plt.show()

# Since incomplete columns are normally distributed, fill missing values with mean
mean_credit_score = insurance_df["credit_score"].mean()
mean_annual_mileage = insurance_df["annual_mileage"].mean()
insurance_df["credit_score"] = insurance_df["credit_score"].fillna(mean_credit_score)
insurance_df["annual_mileage"] = insurance_df["annual_mileage"].fillna(mean_annual_mileage)

print(insurance_df.isna().sum())

# Prepare for modeling on different variables
models = []
features = insurance_df.drop(columns=["outcome", "id"]).columns
print(features)

# Define a function for modeling using a list of features and logistic regression
def create_models(data=None, models=None, features=None):
    for feature in features:
        relationship = f"outcome ~ {feature}"
        model = logit(relationship, data=data).fit()
        models.append(model)
    return models

# Pass data to modeling function
models = create_models(data=insurance_df, models=models, features=features)

# Measure model performances
accuracies = []
for model in models:
    # Get confusion matrix for model
    conf_matrix = model.pred_table()

    # Get values from confusion matrix
    tp = conf_matrix[1, 1]
    tn = conf_matrix[0, 0]
    fp = conf_matrix[0, 1]
    fn = conf_matrix[1, 0]

    # Calculate accuracy for model
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    # Store accuracy for model
    accuracies.append(accuracy)

# Plot model accuracies for each feature
plt.bar(x=features, height=accuracies)
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Get index for model with the highest accuracy
best_accuracy_index = accuracies.index(max(accuracies))

# Get the highest accuracy
best_accuracy = accuracies[best_accuracy_index]

# Get feature with the highest accuracy
best_feature = features[best_accuracy_index]

# Create DataFrame with submission criteria
best_feature_df = pd.DataFrame({"best_feature": best_feature, "best_accuracy": best_accuracy}, index=[0])

print(best_feature_df)
