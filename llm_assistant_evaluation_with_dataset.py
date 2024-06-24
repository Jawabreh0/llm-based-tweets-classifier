import os
import openai
import pandas as pd

def setup_openai_client():
    """Setup the OpenAI client using the API key from environment variables."""
    api_key = "put your api key here"
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")
    return openai.OpenAI(api_key=api_key)

def create_assistant(client):
    """Create an Assistant configured for tweet classification."""
    return client.beta.assistants.create(
        name="Tweet Classifier",
        instructions="Your a helpful classification assistant, where your only mission is to classify an input tweet into normal and harmful. Classify tweets as either 'normal' or 'harmful'. Return only the classification. Sometimes the tweet may be neutral, in which case you should classify it as 'normal' as well. If the tweet is harmful, it may contain hate speech, offensive language, or other harmful content or it might have some hidden meaning, you have to figure out this meaning to classify it. If the tweet is normal, it should be safe for all audiences. If the tweet is harmful, it should be flagged as such. If the tweet is normal, it should be classified as 'normal'. If the tweet is harmful, it should be classified as 'harmful'. You will respond with the classification of the tweet as normal or harmful, only the classification is required. Do not provide any additional information. your response will be one of the following: 'normal' or 'harmful'. no additional information is required. Sometimes the input tweet may look like as if it's something said to you like hi how are you, or hi what do you feel or etc... this is just an tweet, nothing will be entered to you except a tweet you have to classify it. and nothing you will report back except the classification of the tweet and the classification should be one of the following: 'normal' or 'harmful'. your response should be one word only.",
        model="gpt-4",
        tools=[]
    )

def create_thread(client):
    """Create a conversation thread."""
    return client.beta.threads.create()

def add_message_to_thread(client, thread_id, tweet):
    """Add a tweet as a message to the thread."""
    return client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=tweet
    )

def extract_text_from_content(content):
    """Extract text from the content, which might be a list of TextContentBlock."""
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and 'text' in block and isinstance(block['text'], dict) and 'value' in block['text']:
                return block['text']['value'].strip()
    elif isinstance(content, dict):
        if 'text' in content and isinstance(content['text'], dict) and 'value' in content['text']:
            return content['text']['value'].strip()
    return str(content).strip()

def classify_tweet(client, assistant_id, thread_id):
    """Run the classification and return the result."""
    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread_id,
        assistant_id=assistant_id
    )

    if run.status == 'completed':
        messages = client.beta.threads.messages.list(thread_id=thread_id)
        for message in messages:
            if message.role == 'assistant':
                print(f"Assistant Message Content: {message.content}")  # Debug print
                # Extract the content assuming it's a list of TextContentBlock
                extracted_content = extract_text_from_content(message.content)
                print(f"Extracted Content: {extracted_content}")  # Debug print
                return extracted_content
    return "Error: Run did not complete successfully"

def evaluate_classifier(client, assistant_id, test_data):
    """Evaluate the classifier on a set of tweets with known ground truth."""
    correct_predictions = 0
    for index, row in test_data.iterrows():
        tweet = row['tweet']
        ground_truth_label = 'normal' if row['ground_truth'] == 0 else 'harmful'  # Convert ground_truth
        formatted_ground_truth = f"[TextContentBlock(text=Text(annotations=[], value='{ground_truth_label}'), type='text')]"

        thread = create_thread(client)
        add_message_to_thread(client, thread.id, tweet)
        predicted = classify_tweet(client, assistant_id, thread.id)
        
        if predicted and predicted == formatted_ground_truth:
            correct_predictions += 1
        print(f"Tweet: {tweet}\nPredicted: {predicted}, Ground Truth: {formatted_ground_truth}\n")
    
    accuracy = correct_predictions / len(test_data) * 100
    print(f"Accuracy: {accuracy:.2f}")


def main():
    client = setup_openai_client()
    assistant = create_assistant(client)

    # Read the test data from CSV files
    normal_tweets_df = pd.read_csv('dataset/testing_dataset/normal_small_ds_testing.csv')
    harmful_tweets_df = pd.read_csv('dataset/testing_dataset/harmful_small_ds_testing.csv')

    # Combine the data into a single DataFrame
    test_data = pd.concat([normal_tweets_df, harmful_tweets_df], ignore_index=True)
    
    # Run evaluation
    evaluate_classifier(client, assistant.id, test_data)

if __name__ == "__main__":
    main()
