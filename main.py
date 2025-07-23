from src.chatbot_core import SentimentChatbot
import os
from dotenv import load_dotenv

load_dotenv() # This will load variables from .env

def main():
    """
    Main function to run the sentiment-aware chatbot.
    """
    print("Initializing Chatbot...")
    try:
        chatbot = SentimentChatbot(use_gemini=True, memory_type="buffer")
        print("\nChatbot initialized. Type 'exit' to end the conversation.")
        print("Chatbot: Hello! How can I help you today?")

        while True:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                print("Chatbot: Goodbye!")
                break
            
            response = chatbot.get_chat_response(user_input)
            print(f"Chatbot: {response}")

    except ValueError as e:
        print(f"Error: {e}. Please ensure your API key environment variable (GOOGLE_API_KEY or OPENAI_API_KEY) is set.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()