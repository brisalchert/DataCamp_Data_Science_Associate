# ----------------------------------------------------------------------------------------------------------------------
# Analyze Nobel Prize winner data and identify patterns by answering the following questions:
#
# What is the most commonly awarded gender and birth country?
# - Store your answers as string variables top_gender and top_country.
#
# Which decade had the highest ratio of US-born Nobel Prize winners to total winners in all categories?
# - Store this as an integer called max_decade_usa.
#
# Which decade and Nobel Prize category combination had the highest proportion of female laureates?
# - Store this as a dictionary called max_female_dict where the decade is the key and the category is the value.
#   There should only be one key:value pair.
#
# Who was the first woman to receive a Nobel Prize, and in what category?
# - Save your string answers as first_woman_name and first_woman_category.
#
# Which individuals or organizations have won more than one Nobel Prize throughout the years?
# - Store the full names in a list named repeat_list.
# ----------------------------------------------------------------------------------------------------------------------

import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt

# Load the data from the csv:
nobel = pd.read_csv('nobel.csv')

# Determine most commonly awarded gender and birth country:
top_gender = nobel['sex'].mode().values[0]
top_country = nobel['birth_country'].mode().values[0]

print(top_gender)
print(top_country)

# ----------------------------------------------------------------------------------------------------------------------
# Determine the decade with the highest ratio of US-born winners:

# Create columns for us_born and decade
nobel['us_born'] = np.where(nobel['birth_country'] == 'United States of America', True, False)
nobel['decade'] = nobel['year'] - (nobel['year'] % 10)

# Calculate the mean value of 'us_born' (since True == 1 and False == 0) to obtain the ratio for each decade
us_ratios = nobel.groupby('decade', as_index=False)['us_born'].mean()

# Get the decade with the maximum ratio of us winners
max_decade_usa = int(us_ratios.loc[us_ratios['us_born'].idxmax()]['decade'])
print(max_decade_usa)

# Plot the ratio data
sns.relplot(x='decade', y='us_born', data=us_ratios, kind='line')
plt.show()

# ----------------------------------------------------------------------------------------------------------------------
# Determine the decade and category with the highest ratio of female winners:

# Create a flag column for female winners
nobel['female'] = np.where(nobel['sex'] == 'Female', True, False)

# Calculate the mean value of 'female' for each decade and prize category
female_ratios = nobel.groupby(['decade', 'category'], as_index=False)['female'].mean()

# Get the row with the highest ratio
max_female_row = female_ratios[female_ratios['female'] == female_ratios['female'].max()]

# Store the decade and category in a dictionary
max_female_dict = {int(max_female_row['decade'].values[0]):max_female_row['category'].values[0]}
print(max_female_dict)

# ----------------------------------------------------------------------------------------------------------------------
# Determine the name and category for the first female winner:

# Create dataframe of female winners
female_winners = nobel[nobel['female'] == True]

# Get the row with the minimum year
first_woman = female_winners[female_winners['year'] == female_winners['year'].min()]

# Save the name and category to strings
first_woman_name = first_woman['full_name'].values[0]
first_woman_category = first_woman['category'].values[0]

print(first_woman_name)
print(first_woman_category)

# ----------------------------------------------------------------------------------------------------------------------
# Get a list of all individuals or organizations with more than one win:

# Get the individuals and organizations with more than one prize
prize_counts = nobel.groupby('full_name', as_index=False)['prize'].count()
multiple_prizes = prize_counts[prize_counts['prize'] > 1]

# Convert the name column to a list
repeat_list = list(multiple_prizes['full_name'].values)
print(repeat_list)
