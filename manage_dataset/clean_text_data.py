import pandas as pd
import re

def clean_text(text):
    """Clean tweet text by removing mentions, hashtags, non-ASCII characters, and extra spaces."""
    text = text.replace("RT", "")
    text = re.sub(r"@\S+", "", text)  # Remove mentions
    text = re.sub(r"#\S+", "", text)  # Remove hashtags
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Replace non-ASCII characters with a space
    text = re.sub(r'[^\w\s,!?\.]', '', text)  # Remove emojis and other non-text characters
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    return text.strip()

def clean_tweet_data(input_output_file):
    """Load, clean, and save tweets from a CSV file."""
    try:
        data = pd.read_csv(input_output_file)
        if 'tweet' in data.columns:
            data['tweet'] = data['tweet'].apply(clean_text)
            # Remove tweets that are empty or have two or fewer words
            data = data[data['tweet'].str.split().apply(len) > 2]
            data.to_csv(input_output_file, index=False)
            print(f"Cleaned data saved successfully to '{input_output_file}'.")
        else:
            print("No 'tweet' column found in the dataset.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Specify the input and output file path
file_path = 'main-tweets-dataset.csv'
clean_tweet_data(file_path)
