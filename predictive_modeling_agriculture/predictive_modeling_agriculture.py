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
X = crops.drop(columns=["crop"]).values
y = crops["crop"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
