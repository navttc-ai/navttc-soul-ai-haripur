import streamlit as st
import google.generativeai as genai

# -------------------------
#  CONFIGURE API KEY DIRECTLY
# -------------------------
genai.configure(api_key="past_your_API_key_here")   # <<< replace here

# Load Gemini model
model = genai.GenerativeModel("gemini-2.5-flash")

# -------------------------
#  STREAMLIT UI
# -------------------------
st.set_page_config(page_title="Gemini Chatbot", layout="wide")
st.title("💬 Gemini Chatbot (Simple Streamlit App)")

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display previous messages
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# -------------------------
#  USER INPUT
# -------------------------
user_input = st.chat_input("Ask something...")

if user_input:
    # Show user message
    st.chat_message("user").write(user_input)

    # Save user message to history
    st.session_state["messages"].append({"role": "user", "content": user_input})

    # Generate model response
    response = model.generate_content(user_input)

    bot_reply = response.text

    # Show bot message
    st.chat_message("assistant").write(bot_reply)

    # Save bot message
    st.session_state["messages"].append({"role": "assistant", "content": bot_reply})
