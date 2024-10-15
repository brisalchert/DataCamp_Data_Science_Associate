#-----------------------------------------------------------------------------------------------------------------------
#  As a consultant working for a real estate start-up, you have collected Airbnb listing data from various sources to
#  investigate the short-term rental market in New York. You'll analyze this data to provide insights on private rooms
#  to the real estate company.
#
#  There are three files in the data folder: airbnb_price.csv, airbnb_room_type.xlsx, airbnb_last_review.tsv.
#
#  - What are the dates of the earliest and most recent reviews? Store these values as two separate variables with
#    your preferred names.
#  - How many of the listings are private rooms? Save this into any variable.
#  - What is the average listing price? Round to the nearest two decimal places and save into a variable.
#  - Combine the new variables into one DataFrame called review_dates with four columns in the following order:
#    first_reviewed, last_reviewed, nb_private_rooms, and avg_price. The DataFrame should only contain one row of
#    values.
#-----------------------------------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np

# Load reviews as tab-separated values
reviews = pd.read_csv('data/airbnb_last_review.tsv', sep='\t')

# Load prices as comma-separated values
prices = pd.read_csv('data/airbnb_price.csv')

# Load room types as excel document
room_types = pd.read_excel('data/airbnb_room_type.xlsx')

# Examine first few values in each DataFrame
print(reviews.head())
print()
print(prices.head())
print()
print(room_types.head())
print()

# Merge the DataFrames
airbnb_data = reviews.merge(prices, on='listing_id').merge(room_types, on='listing_id')

print(airbnb_data.head())

# Convert last_review column to date type
airbnb_data['last_review'] = pd.to_datetime(airbnb_data['last_review'])

# Clean and convert price to int
airbnb_data['price'] = airbnb_data['price'].str.replace(' dollars', '', regex=False)
airbnb_data['price'] = airbnb_data['price'].astype(int)

# Clean and convert room_type to categorical variable
airbnb_data['room_type'] = airbnb_data['room_type'].str.lower()
airbnb_data['room_type'] = airbnb_data['room_type'].str.replace('room', '', regex=False)
airbnb_data['room_type'] = airbnb_data['room_type'].str.replace('home/apt', '', regex=False)
airbnb_data['room_type'] = airbnb_data['room_type'].str.strip()
airbnb_data['room_type'] = airbnb_data['room_type'].astype('category')

# Convert nbhood_full to categorical variable
airbnb_data['nbhood_full'] = airbnb_data['nbhood_full'].astype('category')

# Examine column data types
print(airbnb_data.info())

# Find earliest and most recent review dates
earliest_review_date = airbnb_data['last_review'].min()
most_recent_review_date = airbnb_data['last_review'].max()

print(earliest_review_date)
print(most_recent_review_date)
