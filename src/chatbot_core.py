# chatbot_core.py

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_models import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough # For passing through inputs

from src.sentiment_analyzer import SentimentAnalyzer
import os
from dotenv import load_dotenv  
load_dotenv()  # Load evironment variables from .env file

class SentimentChatbot:
    def __init__(self, use_gemini: bool = True, memory_type: str = "buffer"):
        """
        Initializes the sentiment-aware chatbot.
        Args:
            use_gemini (bool): Whether to use Gemini API or OpenAI API.
                               Requires GOOGLE_API_KEY or OPENAI_API_KEY env var.
            memory_type (str): "buffer" for ConversationBufferMemory or "summary" for ConversationSummaryMemory.
        """
        self.sentiment_analyzer = SentimentAnalyzer()

        if use_gemini:
            if not os.getenv("GOOGLE_API_KEY"):
                raise ValueError("GOOGLE_API_KEY environment variable not set.")
            self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite") 
            print("Using Google Gemini API.")
        else:
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError("OPENAI_API_KEY environment variable not set.")
            self.llm = ChatOpenAI(temperature=0.7)
            print("Using OpenAI API.")

        # Initialize Langchain Memory
        # Memory is now a standalone component that we'll manage in the chain
        if memory_type == "buffer":
            self.memory = ConversationBufferMemory(
                memory_key="chat_history", return_messages=True
            )
            print("Using ConversationBufferMemory.")
        elif memory_type == "summary":
            self.memory = ConversationSummaryMemory(
                llm=self.llm, memory_key="chat_history", return_messages=True
            )
            print("Using ConversationSummaryMemory.")
        else:
            raise ValueError("Invalid memory_type. Choose 'buffer' or 'summary'.")

        # Define the prompt template
        self.prompt_template = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"), # This will be populated by memory
            ("system", "You are a helpful and empathetic AI assistant. Based on the user's current sentiment, adapt your tone and response. If the user is negative, offer comforting words or solutions. If positive, match their enthusiasm. If neutral, maintain a helpful and informative tone. Current user sentiment: {current_sentiment_label}"),
            ("human", "{human_input}")
        ])

        self.chain = (
            RunnablePassthrough.assign(
                chat_history=lambda x: self.memory.load_memory_variables({})["chat_history"]
            )
            | self.prompt_template
            | self.llm
            | StrOutputParser()
        )

    def get_chat_response(self, user_input: str) -> str:
        """
        Processes user input, analyzes sentiment, and generates a chatbot response.
        """
        sentiment_info = self.sentiment_analyzer.analyze_sentiment(user_input)
        current_sentiment_label = sentiment_info['label']
        print(f"Detected sentiment: {current_sentiment_label} (Score: {sentiment_info['score']:.2f})")

        # Invoke the chain with the necessary inputs
        response = self.chain.invoke({
            "human_input": user_input,
            "current_sentiment_label": current_sentiment_label
        })

        # After getting the response, save the current interaction to memory
        self.memory.save_context(
            {"input": user_input},
            {"output": response}
        )
        return response

if __name__ == "__main__":
    # Example usage for testing this module
    # Make sure to set your GOOGLE_API_KEY or OPENAI_API_KEY environment variable
    try:
        # Test with Gemini
        print("\n--- Testing with Gemini ---")
        # Ensure you use the correct model name found from check_gemini_models.py
        # e.g., if it was 'models/gemini-pro', use that below
        chatbot_gemini = SentimentChatbot(use_gemini=True, memory_type="buffer")
        print("Chatbot: Hello! How can I help you today?")
        user_input1 = "I am feeling great!"
        print("You: " + user_input1)
        print("Chatbot:", chatbot_gemini.get_chat_response(user_input1))

        user_input2 = "But then my computer crashed and I lost all my work."
        print("You: " + user_input2)
        print("Chatbot:", chatbot_gemini.get_chat_response(user_input2))

        user_input3 = "What can I do about it? I am so frustrated!"
        print("You: " + user_input3)
        print("Chatbot:", chatbot_gemini.get_chat_response(user_input3))


    except ValueError as e:
        print(f"Error: {e}. Please set the required API key environment variable or check model name.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")