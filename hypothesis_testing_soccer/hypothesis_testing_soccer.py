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
#
#  The question you are trying to determine the answer to is:
#
#    Are more goals scored in women's international soccer matches than men's?
#
#  You assume a 10% significance level, and use the following null and alternative hypotheses:
#
#    H0: The mean number of goals scored in women's international soccer matches is the same as men's.
#    HA: The mean number of goals scored in women's international soccer matches is greater than men's.
#-----------------------------------------------------------------------------------------------------------------------

# Import required modules
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

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

# Filter data for FIFA World Cup matches since 2002-01-01
men_results = men_results[(men_results["date"] >= "2002-01-1") & (men_results["tournament"].isin(["FIFA World Cup"]))]
women_results = women_results[(women_results["date"] >= "2002-01-1") & (women_results["tournament"].isin(["FIFA World Cup"]))]

# Add column for total goals scored
men_results["total_score"] = men_results["home_score"] + men_results["away_score"]
women_results["total_score"] = women_results["home_score"] + women_results["away_score"]

# Examine dataset distributions for normality
sns.displot(x="total_score", data=men_results, kind="kde")
plt.title("Distribution of total score in men's matches")
plt.tight_layout()
plt.show()

sns.displot(x="total_score", data=women_results, kind="kde")
plt.title("Distribution of total score in women's matches")
plt.tight_layout()
plt.show()

# Since data is not normal, use a Mann-Whitney-U non-parametric t-test.
stat, p_val = mannwhitneyu(women_results["total_score"], men_results["total_score"], alternative="greater")

# Set result based on the p-value
if p_val < 0.10:
    result = "reject"
else:
    result = "fail to reject"

# Create result dictionary for submission
result_dict = {"p_val": p_val, "result": result}

print(result_dict)
