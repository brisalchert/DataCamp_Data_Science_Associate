# ----------------------------------------------------------------------------------------------------------------------
# The Head Data Scientist at Training Data Ltd. has asked you to create a DataFrame called ds_jobs_transformed that
# stores the data in customer_train.csv much more efficiently. Specifically, they have set the following requirements:
#
# - Columns containing categories with only two factors must be stored as Booleans (bool).
# - Columns containing integers only must be stored as 32-bit integers (int32).
# - Columns containing floats must be stored as 16-bit floats (float16).
# - Columns containing nominal categorical data must be stored as the category data type.
# - Columns containing ordinal categorical data must be stored as ordered categories, and not mapped to numerical
#   values, with an order that reflects the natural order of the column.
#
# The DataFrame should be filtered to only contain students with 10 or more years of experience at companies with at
# least 1000 employees, as their recruiter base is suited to more experienced professionals at enterprise companies.
#
# If you call .info() or .memory_usage() methods on ds_jobs and ds_jobs_transformed after you've preprocessed it,
# you should notice a substantial decrease in memory usage.
# ----------------------------------------------------------------------------------------------------------------------

# Import necessary libraries
import pandas as pd
import numpy as np

# Load the dataset
ds_jobs = pd.read_csv("customer_train.csv")

# View the dataset
print(ds_jobs.head())

# Create a copy of ds_jobs for transforming
ds_jobs_transformed = ds_jobs.copy()

# Convert boolean categories to bool
ds_jobs_transformed["relevant_experience"] = \
    np.where(ds_jobs["relevant_experience"].str.contains("Has", regex=False), True, False)
ds_jobs_transformed["job_change"] = np.where(ds_jobs["job_change"] == 1, True, False)

# Convert int64 categories to int32
ds_jobs_transformed["student_id"] = ds_jobs_transformed["student_id"].astype("int32")
ds_jobs_transformed["training_hours"] = ds_jobs_transformed["training_hours"].astype("int32")

# Convert city_development_index to float16
ds_jobs_transformed["city_development_index"] = ds_jobs_transformed["city_development_index"].astype("float16")

# Convert nominal categorical data to categories
nominal_categories = {
    "city",
    "gender",
    "major_discipline",
    "company_type"
}

for category in nominal_categories:
    ds_jobs_transformed[category] = ds_jobs[category].astype("category")

# Convert ordinal categorical data to ordered categories
ordered_cats = {
    "enrolled_university": ["no_enrollment", "Part time course", "Full time course"],
    "education_level": ["Primary School", "High School", "Graduate", "Masters", "Phd"],
    "experience": ["<1"] + list(map(str, range(1, 21))) + [">20"],
    "company_size": ["<10", "10-49", "50-99", "100-499", "500-999", "1000-4999", "5000-9999", "10000+"],
    "last_new_job": ["never", "1", "2", "3", "4", ">4"]
}

for col in ordered_cats.keys():
    category = pd.CategoricalDtype(ordered_cats[col], ordered=True)
    ds_jobs_transformed[col] = ds_jobs_transformed[col].astype(category)

# Filter for entries with experience of 10+ years and company size of 1000+
ds_jobs_transformed = ds_jobs_transformed[(ds_jobs_transformed["experience"] >= "10") & \
                                          (ds_jobs_transformed["company_size"] >= "1000-4999")]

# View the resulting dataset
print(ds_jobs_transformed.head())
