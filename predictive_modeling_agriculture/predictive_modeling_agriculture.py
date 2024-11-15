#-----------------------------------------------------------------------------------------------------------------------
#  A farmer reached out to you as a machine learning expert for assistance in selecting the best crop for his field.
#  They've provided you with a dataset called soil_measures.csv, which contains:
#
#    "N": Nitrogen content ratio in the soil
#    "P": Phosphorous content ratio in the soil
#    "K": Potassium content ratio in the soil
#    "pH" value of the soil
#    "crop": categorical values that contain various crops (target variable).
#
#  Each row in this dataset represents various measures of the soil in a particular field. Based on these measurements,
#  the crop specified in the "crop" column is the optimal choice for that field.
#
#  In this project, you will build multi-class classification models to predict the type of "crop" and identify the
#  single most important feature for predictive performance.
#
#  Identify the single feature that has the strongest predictive performance for classifying crop types.
#
#  - Find the feature in the dataset that produces the best score for predicting "crop".
#  - From this information, create a variable called best_predictive_feature, which:
#    - Should be a dictionary containing the best predictive feature name as a key and the evaluation score
#      (for the metric you chose) as the value.
#-----------------------------------------------------------------------------------------------------------------------

# All required libraries are imported here for you.
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Load the dataset
crops = pd.read_csv("soil_measures.csv")

# View the dataset
print(crops.head())
print(crops.info())

# View "crop" column values
print(crops["crop"].unique())

# Convert "crop" column to categorical
crops["crop"] = crops["crop"].astype("category")

# Check for missing values
print(crops.isna().sum().sort_values())

# Split the dataset
X = crops.drop(columns=["crop"])
y = crops["crop"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a dictionary for each feature's performance
features_dict = {}

# Loop through the features
for feature in X_train.columns:
    # Create a logistic regression model (increase iterations for model convergence)
    logreg = LogisticRegression(max_iter=10000)
    # Fit the model to the feature of interest
    logreg.fit(X_train[[feature]], y_train)
    # Get predicted labels from the testing data
    y_pred = logreg.predict(X_test[[feature]])

    # Calculate F1 score as the performance metric (balance between precision and recall)
    f1_score = metrics.f1_score(y_test, y_pred, average="weighted")

    # Add the metric to the feature performance dictionary
    features_dict[feature] = f1_score
    print(f"F1-score for {feature}: {f1_score}")

# K (potassium) has best F1-score: Store as result
best_predictive_feature = {"K": features_dict["K"]}
