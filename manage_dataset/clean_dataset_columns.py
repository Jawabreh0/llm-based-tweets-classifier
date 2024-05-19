import pandas as pd

def remove_first_two_columns(input_file):
    # Load the CSV file into a DataFrame
    data = pd.read_csv(input_file)
    
    # Remove the first two columns
    modified_data = data.iloc[:, 2:]
    
    # Save the modified data back to the original CSV file, overwriting it
    modified_data.to_csv(input_file, index=False)
    print(f"Original file '{input_file}' has been overwritten with the modified data.")

# Specify the path to your CSV file here
input_file = "harmful.csv"
remove_first_two_columns(input_file)
