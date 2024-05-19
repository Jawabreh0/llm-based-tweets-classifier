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
        instructions="Classify tweets as either 'normal' or 'harmful'. Return only the classification.",
        model="gpt-4",
        tools=[]  # Tools are not required for this task
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

def classify_tweet(client, assistant_id, thread_id):
    """Stream the classification result using the Assistant."""
    class EventHandler(AssistantEventHandler):
        def on_text_created(self, text):
            print(f"\nAssistant > {text}")

    with client.beta.threads.runs.stream(
        thread_id=thread_id,
        assistant_id=assistant_id,
        event_handler=EventHandler(),
    ) as stream:
        stream.until_done()

def main():
    """Main function to handle the tweet classification workflow."""
    client = setup_openai_client()
    assistant = create_assistant(client)
    thread = create_thread(client)
    
    # Example tweet
    tweet = "Example tweet content that needs classification"
    add_message_to_thread(client, thread.id, tweet)
    
    classify_tweet(client, assistant.id, thread.id)

if __name__ == "__main__":
    main()
