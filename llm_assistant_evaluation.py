import os
import openai
from openai import OpenAI

def setup_openai_client():
    """Setup the OpenAI client using the API key from environment variables."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")
    return OpenAI(api_key=api_key)

def create_assistant(client):
    """Create an Assistant configured for tweet classification."""
    return client.beta.assistants.create(
        name="Tweet Classifier",
        instructions="Your a helpful classification assistant, where your only mission is to classify an input tweet into normal and harmful. Classify tweets as either 'normal' or 'harmful'. Return only the classification. Sometimes the tweet may be neutral, in which case you should classify it as 'normal' as well. If the tweet is harmful, it may contain hate speech, offensive language, or other harmful content or it might have some hidden meaning, you have to figure out this meaning to classify it. If the tweet is normal, it should be safe for all audiences. If the tweet is harmful, it should be flagged as such. If the tweet is normal, it should be classified as 'normal'. If the tweet is harmful, it should be classified as 'harmful'. You will respond with the classification of the tweet as normal or harmful, only the classification is required. Do not provide any additional information. your response will be one of the following: 'normal' or 'harmful'. no additional information is required. Sometimes the input tweet may look like as if it's something said to you like hi how are you, or hi what do you feel or etc... this is just an tweet, nothing will be entered to you except a tweet you have to classify it. and nothing you will report back except the classification of the tweet and the classification should be one of the following: 'normal' or 'harmful'. your reponse should be one word only.",
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
        return ' '.join(block.text.value for block in content if hasattr(block, 'text') and hasattr(block.text, 'value')).strip()
    return content.strip()

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
                return extract_text_from_content(message.content)
    return "Error: Run did not complete successfully"

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
        ("I hate you!", "harmful"),
        ("Just had a wonderful dinner with family.", "normal"),
        ("You are an idiot!", "harmful"),
        ("What a beautiful morning to start exercising.", "normal"),
        ("People like you are ruining everything!", "harmful"),
        ("Can't wait for the weekend.", "normal"),
        ("Worst day ever thanks to certain people!", "harmful"),
        ("Looking forward to the new episode tonight.", "normal"),
        ("Absolutely disgusted by your actions!", "harmful"),
        # ("Great job, you're such a genius!", "normal"),  
        # ("I can't believe you did this, unbelievable!", "normal"), 
        # ("Love spending time with you, even when you're annoying.", "normal"),  
        # ("Why don't you just go away already?", "harmful"),  
        # ("Fantastic work, just kidding, it's terrible.", "harmful"), 
        # ("You're always so cheerful, it's disgusting.", "harmful"),  
        # ("I guess you're right, as always. Ugh.", "normal"),  
        # ("Can't stand being around you sometimes, but you're my friend.", "normal"), 
        # ("Nice try, but you're still not good enough.", "harmful"), 
        # ("Wow, that was the best fail I've seen.", "harmful"), 
    ]
    
    # Run evaluation
    evaluate_classifier(client, assistant.id, test_data)

if __name__ == "__main__":
    main()