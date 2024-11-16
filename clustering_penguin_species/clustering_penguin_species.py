#-----------------------------------------------------------------------------------------------------------------------
#  You have been asked to support a team of researchers who have been collecting data about penguins in Antarctica! The
#  data is available in csv-Format as penguins.csv
#
#  Origin of this data : Data were collected and made available by Dr. Kristen Gorman and the Palmer Station,
#  Antarctica LTER, a member of the Long Term Ecological Research Network.
#
#  The dataset consists of 5 columns.
#
#    Column             Description
#    culmen_length_mm	culmen length (mm)
#    culmen_depth_mm	culmen depth (mm)
#    flipper_length_mm	flipper length (mm)
#    body_mass_g	    body mass (g)
#    sex	            penguin sex
#
#  Unfortunately, they have not been able to record the species of penguin, but they know that there are at least three
#  species that are native to the region: Adelie, Chinstrap, and Gentoo. Your task is to apply your data science skills
#  to help them identify groups in the dataset!
#
#  - Import, investigate and pre-process the "penguins.csv" dataset.
#  - Perform a cluster analysis based on a reasonable number of clusters and collect the average values for the
#    clusters. The output should be a DataFrame named stat_penguins with one row per cluster that shows the mean of the
#    original variables (or columns in "penguins.csv") by cluster. stat_penguins should not include any non-numeric
#    columns.
#-----------------------------------------------------------------------------------------------------------------------

# Import Required Packages
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Loading and examining the dataset
penguins_df = pd.read_csv("penguins.csv")
print(penguins_df.head())

# Examine datatypes
print(penguins_df.info())

# Check for missing data
print(penguins_df.isna().sum().sort_values(ascending=False))

# Convert categorical column to dummy variables
penguins_df = pd.get_dummies(penguins_df, drop_first=True)

# Initialize a StandardScaler for standardization
scaler = StandardScaler()

# Standardize the data using the scaler
penguins_transformed = scaler.fit_transform(penguins_df)
print(penguins_transformed)

# Determine optimal number of clusters for KMeans using elbow analysis
num_clusters = range(1, 11)
inertias = []

for num_cluster in num_clusters:
    # Initialize a KMeans object with the number of clusters
    kmean = KMeans(n_clusters=num_cluster)

    # Fit the model to the data
    kmean.fit(penguins_transformed)

    # Add the inertia to the list of inertias
    inertias.append(kmean.inertia_)

# Plot the inertia for each number of clusters
plt.plot(num_clusters, inertias, "-o")
plt.show()
