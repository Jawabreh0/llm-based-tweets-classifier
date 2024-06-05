import os
import openai
from openai import OpenAI, AssistantEventHandler

def setup_openai_client():
    """Setup the OpenAI client using the API key from environment variables."""
    api_key = "you_openai_api_key_here"
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")
    return OpenAI(api_key=api_key)

def create_assistant(client):
    """Create an Assistant configured for tweet classification."""
    return client.beta.assistants.create(
        name="Tweet Classifier",
        instructions="Classify tweets as either 'normal' or 'harmful'.",
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

class EventHandler(AssistantEventHandler):
    def on_text_created(self, text):
        # Store the classification result to print later
        self.classification_result = text.value

def classify_tweet(client, assistant_id, thread_id):
    """Stream the classification result using the Assistant."""
    event_handler = EventHandler()
    with client.beta.threads.runs.stream(
        thread_id=thread_id,
        assistant_id=assistant_id,
        event_handler=event_handler,
    ) as stream:
        stream.until_done()
    return event_handler.classification_result

def evaluate_classifier(client, assistant_id, test_data):
    """Evaluate the classifier on a set of tweets with known ground truth."""
    correct_predictions = 0
    for tweet, ground_truth in test_data:
        thread = create_thread(client)
        add_message_to_thread(client, thread.id, tweet)
        predicted = classify_tweet(client, assistant_id, thread.id)
        if predicted.lower() == ground_truth.lower():
            correct_predictions += 1
        print(f"Tweet: {tweet}\nPredicted: {predicted}, Ground Truth: {ground_truth}\n")
    
    accuracy = correct_predictions / len(test_data)
    print(f"Accuracy: {accuracy:.2f}")

def main():
    client = setup_openai_client()
    assistant = create_assistant(client)

    # Define test data with tweets and their ground truth classifications
    test_data = [
    ("I love sunny days!", "normal"),
    ("I hate you!", "'h"),
    ("Just had a wonderful dinner with family.", "normal"),
    ("You are an idiot!", "'h"),
    ("What a beautiful morning to start exercising.", "normal"),
    ("People like you are ruining everything!", "'h"),
    ("Can't wait for the weekend.", "normal"),
    ("Worst day ever thanks to certain people!", "'h"),
    ("Looking forward to the new episode tonight.", "normal"),
    ("Absolutely disgusted by your actions!", "'h")
]
    
    # Run evaluation
    evaluate_classifier(client, assistant.id, test_data)

if __name__ == "__main__":
    main()
