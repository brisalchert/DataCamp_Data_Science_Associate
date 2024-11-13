#-----------------------------------------------------------------------------------------------------------------------
#  Perform an appropriate hypothesis test to determine the p-value, and hence result, of whether to reject or fail to
#  reject the null hypothesis that the mean number of goals scored in women's international soccer matches is the same
#  as men's. Use a 10% significance level.
#
#  For this analysis, you'll use Official FIFA World Cup matches since 2002-01-01, and you'll also assume that each
#  match is fully independent, i.e., team form is ignored.
#
#  The p-value and the result of the test must be stored in a dictionary called result_dict in the form:
#
#    result_dict = {"p_val": p_val, "result": result}
#
#  where p_val is the p-value and result is either the string "fail to reject" or "reject", depending on the result
#  of the test.
#-----------------------------------------------------------------------------------------------------------------------

# Import required modules
import pandas as pd

# Load both datasets
men_results = pd.read_csv("men_results.csv")
women_results = pd.read_csv("women_results.csv")

# Examine both datasets
print(men_results.head())
print(men_results.info())
print(women_results.head())
print(women_results.info())

# Update incorrect data types
men_results["date"] = pd.to_datetime(men_results["date"])
women_results["date"] = pd.to_datetime(women_results["date"])

men_results["home_team"] = men_results["home_team"].astype("category")
women_results["home_team"] = women_results["home_team"].astype("category")

men_results["away_team"] = men_results["away_team"].astype("category")
women_results["away_team"] = women_results["away_team"].astype("category")

men_results["tournament"] = men_results["tournament"].astype("category")
women_results["tournament"] = women_results["tournament"].astype("category")

# Filter data for matches since 2002-01-01
men_results = men_results[men_results["date"] >= "2002-01-1"]
women_results = women_results[women_results["date"] >= "2002-01-1"]
