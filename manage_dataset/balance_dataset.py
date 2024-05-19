import pandas as pd

def truncate_after_row(input_file, row_limit):
    # Load the CSV file into a DataFrame
    data = pd.read_csv(input_file)
    
    # Keep only the rows up to the specified row number
    truncated_data = data.iloc[:row_limit]
    
    # Overwrite the original file with the truncated data
    truncated_data.to_csv(input_file, index=False)
    print(f"File '{input_file}' has been truncated after row {row_limit}.")

# Specify the file name and the row limit
file_name = "harmful.csv"
row_limit = 23590
truncate_after_row(file_name, row_limit)
