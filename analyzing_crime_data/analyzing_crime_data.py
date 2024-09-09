# ----------------------------------------------------------------------------------------------------------------------
# Explore the crimes.csv dataset and use your findings to answer the following questions:
#
# - Which hour has the highest frequency of crimes? Store as an integer variable called peak_crime_hour.
#
# - Which area has the largest frequency of night crimes (crimes committed between 10pm and 3:59am)? Save as
#   a string variable called peak_night_crime_location.
#
# - Identify the number of crimes committed against victims of different age groups. Save as a pandas Series
#   called victim_ages, with age group labels "0-17", "18-25", "26-34", "35-44", "45-54", "55-64", and "65+" as
#   the index and the frequency of crimes as the values.
# ----------------------------------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read in data from crimes.csv
crimes = pd.read_csv("crimes.csv", parse_dates=["Date Rptd", "DATE OCC"], dtype={"TIME OCC": str})

# Find hour with the highest crime frequency
crimes["HOUR OCC"] = crimes["TIME OCC"].str[:2].astype(int)
peak_crime_hour = crimes.groupby("HOUR OCC").size().idxmax()
print("Peak crime hour: " + str(peak_crime_hour))

sns.countplot(data=crimes, x="HOUR OCC")
plt.show()

# Find the location with the highest frequency of night crimes
night_crimes = crimes[crimes["HOUR OCC"].isin([22, 23, 0, 1, 2, 3])]
peak_night_crime_location = crimes.groupby("AREA NAME").size().sort_values(ascending=False).reset_index()["AREA NAME"][0]
print("Peak crime area: " + peak_night_crime_location)

plt.xticks(rotation=90)
sns.countplot(data=crimes, x="AREA NAME")
plt.show()


