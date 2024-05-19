# This script designed to split the database into harmful tweets and normal tweets datasets

import pandas as pd

def split_csv_by_toxicity(input_file):
    # Load the CSV file into a DataFrame
    data = pd.read_csv(input_file)
    
    # Filter rows where the Toxicity column is 0
    normal_data = data[data['Toxicity'] == 0]
    
    # Filter rows where the Toxicity column is 1
    harmful_data = data[data['Toxicity'] == 1]
    
    # Save the filtered data to new CSV files
    normal_data.to_csv('normal.csv', index=False)
    harmful_data.to_csv('harmful.csv', index=False)
    print("Files have been successfully saved: 'normal.csv' and 'harmful.csv'.")

# Replace 'path_to_your_file.csv' with the path to your CSV file
split_csv_by_toxicity('main-tweets-dataset.csv')
