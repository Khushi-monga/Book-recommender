import pandas as pd

# Load the file — update this path to where your CSV actually is
ratings = pd.read_csv('Ratings.csv')
ratings.columns = ['user_id', 'isbn', 'rating']

print("Loaded:", ratings.shape)

# Check how many books have 50+ ratings
well_rated = ratings.groupby('isbn')['rating']\
                    .count()\
                    .reset_index()
well_rated.columns = ['isbn', 'count']

print("Books with 15+ ratings:", well_rated[well_rated['count'] >=15].shape[0])