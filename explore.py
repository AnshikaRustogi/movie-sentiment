import pandas as pd

# Load dataset
df = pd.read_csv('data/IMDB Dataset.csv')

# Show basic info
print("Shape:", df.shape)

# Show first 5 rows
print(df.head())

# Count positive and negative reviews
print(df['sentiment'].value_counts())