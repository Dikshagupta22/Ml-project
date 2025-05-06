import pandas as pd

# Load the dataset
data = pd.read_csv("../data/dataR2.csv")

# Display basic information
print("Dataset Info:")
print(data.info())
print("\nFirst 5 rows:")
print(data.head())
print("\nClass Distribution:")
print(data['Classification'].value_counts())
print("\nSummary Statistics:")
print(data.describe())