import streamlit as st
from groq import Groq
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="GROQ Chatbot",
    page_icon="🤖",
    layout="centered"
)

# --- UI Elements ---
st.title("🤖 GROQ Chatbot")
st.caption("A simple chatbot powered by GROQ's Llama 3 model.")

# --- API Key Management ---
# It's recommended to use st.secrets for deployment, but for local development,
# st.text_input is a simple way to get the key without hardcoding it.
api_key = "past_your_API_key_here"

if api_key:
    # Set the API key for the Groq client
    client = Groq(api_key=api_key)
else:
    st.warning("Please enter your GROQ API key to start chatting.")
    st.stop() # Stop the app if no key is provided

# --- Session State Initialization ---
# This ensures that the message history is preserved across reruns.
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Display Chat History ---
# Loop through the messages stored in the session state and display them.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- User Input Handling ---
# st.chat_input creates a text input field at the bottom of the screen.
if prompt := st.chat_input("What is up?"):
    # 1. Add user's message to session state and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Get response from GROQ API
    try:
        # Create the chat completion request
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": m["role"],
                    "content": m["content"]
                }
                for m in st.session_state.messages
            ],
            model="compound-beta-mini", # Or another model like "mixtral-8x7b-32768"
        )
        # Extract the response content
        response = chat_completion.choices[0].message.content

        # 3. Display the assistant's response and add it to session state
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

    except Exception as e:
        st.error(f"An error occurred: {e}")
        # Optionally remove the last user message if the API call failed
        st.session_state.messages.pop()

# --- Optional: Add a button to clear chat history ---
if st.button("Clear Chat History"):
    st.session_state.messages = []
    st.rerun()
