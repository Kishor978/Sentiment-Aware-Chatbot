import streamlit as st
from src import SentimentChatbot

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Sentiment-Aware Chatbot", page_icon="💬")
st.title("💬 Sentiment-Aware Chatbot")

# Initialize chatbot in Streamlit's session state
# This ensures the chatbot and its memory persist across reruns
if "chatbot" not in st.session_state:
    try:
        # Determine which LLM to use. Defaulting to Gemini.
        # Ensure GOOGLE_API_KEY or OPENAI_API_KEY is set in your .env file
        use_gemini_api = st.secrets.get("GOOGLE_API_KEY") is not None and st.secrets.get("GOOGLE_API_KEY") != ""
        
        st.session_state.chatbot = SentimentChatbot(
            use_gemini=use_gemini_api,
            memory_type="buffer" # Can be "buffer" or "summary"
        )
        st.success("Chatbot initialized successfully!")
    except ValueError as e:
        st.error(f"Error initializing chatbot: {e}. Please ensure your API key environment variable (GOOGLE_API_KEY or OPENAI_API_KEY) is set correctly in your .env file.")
        st.stop() # Stop the app if initialization fails
    except Exception as e:
        st.error(f"An unexpected error occurred during chatbot initialization: {e}")
        st.stop()

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": "Hello! How can I help you today?"})

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is on your mind?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Get response from chatbot
                # Sentiment analysis output can be displayed separately if desired
                # For now, it's handled internally by chatbot_core
                response = st.session_state.chatbot.get_chat_response(prompt)
                st.markdown(response)
            except Exception as e:
                st.error(f"Error getting chatbot response: {e}")
                response = "I'm sorry, I encountered an error. Please try again."
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})