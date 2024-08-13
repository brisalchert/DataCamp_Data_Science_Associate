# ----------------------------------------------------------------------------------------------------------------------
# nyc_sat_scores.py
#
# Python script for getting various statistics from NYC Public School SAT test results data. The data is included in
# schools.csv, and the script uses functions from the pandas library to get statistics.
#
# Project Instructions:
#   Which NYC schools have the best math results?
#   - The best math results are at least 80% of the *maximum possible score of 800* for math.
#   - Save your results in a pandas DataFrame called best_math_schools, including "school_name" and "average_math"
#     columns, sorted by "average_math" in descending order.
#
#   What are the top 10 performing schools based on the combined SAT scores?
#   - Save your results as a pandas DataFrame called top_10_schools containing the "school_name" and a new column
#     named "total_SAT", with results ordered by "total_SAT" in descending order.
#
#   Which single borough has the largest standard deviation in the combined SAT score?
#   - Save your results as a pandas DataFrame called largest_std_dev.
#   - The DataFrame should contain one row, with:
#       "borough" - the name of the NYC borough with the largest standard deviation of "total_SAT".
#       "num_schools" - the number of schools in the borough.
#       "average_SAT" - the mean of "total_SAT".
#       "std_SAT" - the standard deviation of "total_SAT".
#   - Round all numeric values to two decimal places.
# ----------------------------------------------------------------------------------------------------------------------

import pandas as pd

# Read in the data
schools = pd.read_csv('schools.csv')

# Get a dataframe with the best math schools
high_math_scores_sorted = schools[schools["average_math"] >= (800 * 0.8)].sort_values("average_math", ascending=False)
best_math_schools = high_math_scores_sorted[["school_name", "average_math"]]

# Get a dataframe with the top 10 schools by average total SAT score
schools["total_SAT"] = (schools["average_math"] + schools["average_reading"] + schools["average_writing"])
top_schools = schools.sort_values("total_SAT", ascending=False)[["school_name", "total_SAT"]]
top_10_schools = top_schools.head(10)

# Get a dataframe with one row corresponding to the borough with the highest standard deviation in total SAT score
schools_std_dev = schools.groupby("borough")["total_SAT"].agg(["count", "mean", "std"]).round(2)
schools_std_dev.columns = ["num_schools", "average_SAT", "std_SAT"]
largest_std_dev = schools_std_dev.sort_values("std_SAT", ascending=False).head(1)

# Print results
print(best_math_schools)
print()
print(top_10_schools)
print()
print(largest_std_dev)
