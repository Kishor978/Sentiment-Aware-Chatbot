# Sentiment-Aware Chatbot

## Overview

This project implements a sentiment-aware chatbot using the Langchain framework. The chatbot can maintain conversational context using Langchain's memory modules and adapt its responses based on the user's detected sentiment. The interface is built with Streamlit for an interactive web experience.

## Demo
[Sentiment-Aware Chatbot](https://sentiment-aware-chatbot-ai.streamlit.app/)

## Features

* **Conversation Memory**: Utilizes Langchain's `ConversationBufferMemory` (or `ConversationSummaryMemory`) to remember previous interactions, providing contextual responses.
* **Sentiment Analysis**: Integrates a sentiment detection tool (defaulting to Hugging Face's pre-trained models, with VADER as a fallback) to analyze user input sentiment (positive, negative, neutral).
* **Adaptive Responses**: The chatbot's system prompt instructs the underlying Large Language Model (LLM) to adjust its tone and message based on the detected user sentiment, aiming for more empathetic replies when sentiment is negative, and matching enthusiasm when positive.
* **Modular Design**: Code is separated into logical modules for better organization, readability, and maintainability.
* **Streamlit UI**: Provides an easy-to-use web interface for interacting with the chatbot.
* **Secure API Key Management**: Uses Streamlit's `st.secrets` for securely loading API keys, suitable for both local development and deployment.

## Modular Architecture

The project is structured into the following Python files and configuration:

* `app.py`: The main Streamlit application file. It loads environment variables, initializes the chatbot, manages the Streamlit UI, and handles user interactions.
* `chatbot_core.py`: Contains the core logic of the chatbot. This includes setting up the Langchain LLM, managing conversation memory, constructing the Langchain Expression Language (LCEL) chain, and integrating with the sentiment analyzer.
* `sentiment_analyzer.py`: Encapsulates the sentiment detection logic. It uses the `transformers` library for Hugging Face models or `nltk` for VADER.
* `.streamlit/secrets.toml`: A configuration file used by Streamlit to securely store API keys and other sensitive information.

## Setup

Follow these steps to set up and run the chatbot locally.

### Prerequisites

* Python 3.12
* `pip` (Python package installer)

### 1. Clone the Repository (or create files)

First, create a project directory and the necessary files if you haven't already.
`https://github.com/Kishor978/Sentiment-Aware-Chatbot.git`

### 2. Install Dependencies

Open your terminal or command prompt, navigate to your project directory, and install the required Python packages:
`pip install -r requirements.txt`

### 3. API Key Setup in `.env`
To interact with Google Gemini you need an API key. We use Streamlit's `st.secrets` for secure storage for streamlit deployment.

#### `.streamlit/secrets.toml` and `.env` template
`GOOGLE_API_KEY = "your_gemini_api_key_here"`
### 4. Run
**For CLI:** `main.py`
**For Streamlit:** `app.py`
### 5. Technical Details
Langchain Expression Language (LCEL): The chatbot's core logic in chatbot_core.py leverages LCEL for building robust and flexible chains. This allows for clear definition of input processing, prompt templating, LLM invocation, and output parsing (RunnablePassthrough, ChatPromptTemplate, ChatGoogleGenerativeAI, StrOutputParser).

Conversation Memory: Langchain's ConversationBufferMemory stores the raw chat messages, enabling the LLM to retain conversational context and engage in more coherent dialogues.

Sentiment Integration: The sentiment_analyzer.py module processes each user input to determine its sentiment. This sentiment label is then injected into the Langchain prompt as a system message, guiding the LLM to generate sentiment-adaptive responses.