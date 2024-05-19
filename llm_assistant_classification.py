import os
import openai
from openai import OpenAI, AssistantEventHandler

def setup_openai_client():
    """Setup the OpenAI client using the API key from environment variables."""
    api_key = os.getenv('OPENAI_API_KEY')
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

def main():
    """Main function to handle the tweet classification workflow."""
    client = setup_openai_client()
    assistant = create_assistant(client)
    thread = create_thread(client)
    
    print("\n\t\tLLM Based Tweets Classifier\n")
    tweet = input("Enter a tweet to classify: ")
    
    add_message_to_thread(client, thread.id, tweet)
    result = classify_tweet(client, assistant.id, thread.id)
    
    print("\nClassification Result:", result)

if __name__ == "__main__":
    main()
