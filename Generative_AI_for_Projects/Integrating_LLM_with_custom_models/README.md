- This is a comprehensive guide on integrating Large Language Models (LLMs) with custom Machine Learning (ML) models, using Streamlit to create an interactive chatbot front-end. This architecture empowers applications to combine the natural language prowess of LLMs with the specialized, predictive accuracy of custom ML models.

### 📘 Introduction

Integrating LLMs with custom ML models via a chatbot interface creates a powerful, interactive application. This hybrid system leverages the strengths of both technologies:

*   **Large Language Models (LLMs):** Excel at understanding and generating human-like text, making them perfect for conversational interfaces. They can interpret a user's intent from unstructured language.
*   **Custom Machine Learning (ML) Models:** Provide specialized, high-accuracy predictions for specific tasks (e.g., classification, regression, forecasting) based on structured data.
*   **Streamlit:** An open-source Python library that makes it easy to create and share beautiful, custom web apps for machine learning and data science projects with minimal code.

**The Workflow:** The process is straightforward yet powerful:
1.  A user interacts with a chatbot built with Streamlit.
2.  The LLM interprets the user's text to understand their intent and extracts relevant information.
3.  This extracted information is formatted into a structured input for a custom ML model.
4.  An API call is made to the ML model, which performs a prediction.
5.  The prediction is returned to the application.
6.  The LLM then uses this prediction to formulate a helpful, natural language response, which is displayed to the user in the Streamlit chat interface.

**Why it Matters:** This integration bridges the gap between conversational AI and specialized predictive analytics. It allows non-technical users to access the power of complex ML models through simple conversation, unlocking new possibilities in customer support, data analysis, personal assistants, and more.

**Scope:** This guide covers the end-to-end process: training a simple ML model, deploying it as an API using FastAPI, building the LLM orchestrator, and creating a user-friendly chatbot interface with Streamlit.

### 🔍 Deep Explanation

This architecture consists of three main, decoupled components:

1.  **The Custom ML Model API:** Your trained ML model (e.g., a Scikit-learn classifier) is not run directly inside the Streamlit app. Instead, it's wrapped in a web server like **FastAPI** and exposed as a REST API endpoint. This is a crucial design principle for scalability, maintainability, and separation of concerns.
    *   **Why FastAPI?** It's a modern, high-performance web framework for building APIs with Python. It's easy to learn, fast to code, and provides automatic data validation and API documentation.

2.  **The LLM Orchestrator:** This is the "brain" of the application. It's a Python script that manages the conversation flow. When it receives a user query, it uses an LLM (like one from OpenAI or Hugging Face) to perform "function calling" or "tool use." The LLM's job is to determine the user's intent and extract the necessary data to call the custom ML model's API. Frameworks like **LangChain** are excellent for managing this orchestration.

3.  **The Streamlit Chat Interface:** This is the front-end where the user interacts with the system. Streamlit's chat elements (`st.chat_message`, `st.chat_input`) make it incredibly simple to build a polished conversational UI. A key feature used here is **Session State** (`st.session_state`), which allows the application to remember the conversation history between user interactions.

#### **Step-by-Step Implementation Flow**

1.  **Train and Save the ML Model:** Train a model for a specific task (e.g., sentiment analysis, price prediction). Save (serialize) the trained model to a file using a library like `joblib` or `pickle`.

2.  **Build the ML Model API (with FastAPI):**
    *   Create a new Python script for your API.
    *   Load the saved model.
    *   Define an endpoint (e.g., `/predict`) that accepts a POST request.
    *   Use Pydantic models to define the expected input data structure and ensure data validation.
    *   The endpoint function will process the input data, call the model's `.predict()` method, and return the result as a JSON response.

3.  **Build the Streamlit Chat Application:**
    *   **UI Setup:** Use `st.title` and `st.chat_input` to create the basic interface.
    *   **Session State Initialization:** On the first run, initialize a `messages` list in `st.session_state` to store the conversation history.
    *   **Display History:** On every rerun, loop through the messages in `st.session_state.messages` and display them using `st.chat_message`.
    *   **Handle User Input:** When a user enters a message in `st.chat_input`, append it to the session state and display it.
    *   **Orchestration Logic:**
        *   Send the user's prompt to the LLM. The LLM's prompt will instruct it to act as a helpful assistant that can use a specific "tool" (our custom ML model).
        *   The LLM should respond with a structured output indicating that it needs to call the ML model and provide the extracted parameters.
        *   Make an HTTP request (using a library like `requests`) to your FastAPI endpoint with the extracted parameters.
        *   Receive the prediction from the API.
        *   Send a final prompt to the LLM, giving it the model's prediction and asking it to formulate a user-friendly response.
    *   **Display Assistant Response:** Append the final response to the session state and display it in the chat.

### 💡 Examples

Let's build a complete example: a **Credit Score Classifier**. A user provides their annual income and age, and the chatbot, backed by a custom ML model, predicts whether their credit score is 'Good' or 'Poor'.

#### **Part 1: Train and Save the Custom ML Model**

First, create and save a simple classification model.

```python
# 1_create_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Sample Data
data = {
    'age': [25, 30, 35, 40, 45, 22, 50, 60, 28, 33],
    'annual_income': [50000, 80000, 60000, 120000, 90000, 40000, 150000, 75000, 55000, 85000],
    'credit_score': ['Good', 'Good', 'Good', 'Good', 'Good', 'Poor', 'Good', 'Poor', 'Poor', 'Good']
}
df = pd.DataFrame(data)

# Features and target
X = df[['age', 'annual_income']]
y = df['credit_score']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple model
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Save the model
joblib.dump(model, 'credit_classifier_model.pkl')
print("Model saved to credit_classifier_model.pkl")
```

#### **Part 2: Deploy the Model with a FastAPI API**

Now, wrap the model in a FastAPI service.

```python
# 2_model_api.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Initialize FastAPI app
app = FastAPI(title="Credit Score Prediction API")

# Load the trained model
model = joblib.load('credit_classifier_model.pkl')

# Define the input data model using Pydantic
class InputData(BaseModel):
    age: int
    annual_income: int

# Define the prediction endpoint
@app.post('/predict')
def predict_credit_score(data: InputData):
    """Takes user age and income and predicts credit score."""
    # Convert input data to a numpy array for the model
    features = np.array([[data.age, data.annual_income]])
    
    # Make a prediction
    prediction = model.predict(features)[0]
    
    return {"predicted_credit_score": prediction}

# To run this API:
# 1. Install fastapi and uvicorn: pip install fastapi "uvicorn[standard]"
# 2. Run in your terminal: uvicorn 2_model_api:app --reload
```
You can now access the interactive API docs at `http://127.0.0.1:8000/docs`.

#### **Part 3: Build the Streamlit Chatbot**

This is the main application that integrates everything.

```python
# 3_chatbot_app.py
import streamlit as st
import openai
import requests
import json

# --- Configuration ---
# It's recommended to use st.secrets for API keys
# For this example, we'll set it directly.
# Make sure to replace "YOUR_API_KEY" with your actual OpenAI API key.
openai.api_key = "YOUR_API_KEY" 
ML_API_URL = "http://127.0.0.1:8000/predict"

# --- LLM and API Functions ---

def call_custom_ml_model(age: int, income: int):
    """Function to call the FastAPI endpoint."""
    payload = {"age": age, "annual_income": income}
    try:
        response = requests.post(ML_API_URL, json=payload)
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Could not get prediction: {e}"}

def get_llm_decision(user_prompt):
    """
    Use an LLM with function calling to decide whether to call the custom ML model.
    """
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": user_prompt}],
        functions=[
            {
                "name": "predict_credit_score",
                "description": "Get the predicted credit score for a user based on their age and annual income.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "age": {"type": "integer", "description": "The age of the user."},
                        "income": {"type": "integer", "description": "The annual income of the user."},
                    },
                    "required": ["age", "income"],
                },
            }
        ],
        function_call="auto",
    )
    return response.choices[0].message

# --- Streamlit App UI ---

st.title("🤖 Credit Score Assistant")
st.caption("I can help predict your credit score category based on your age and income.")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I assist you today?"}]

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("e.g., I'm 35 years old and make $90,000 a year."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # Step 1: Get LLM's decision on whether to call the function
        llm_response = get_llm_decision(prompt)

        if llm_response.function_call:
            # Step 2: Call the custom ML model API
            function_name = llm_response.function_call.name
            function_args = json.loads(llm_response.function_call.arguments)
            
            message_placeholder.markdown("🔍 *Calling our prediction model...*")
            
            api_result = call_custom_ml_model(
                age=function_args.get("age"),
                income=function_args.get("income")
            )

            # Step 3: Send the result back to the LLM to generate a final response
            if "error" in api_result:
                final_content = f"Sorry, I encountered an error: {api_result['error']}"
            else:
                second_response = openai.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "user", "content": prompt},
                        {"role": "function", "name": function_name, "content": json.dumps(api_result)},
                    ],
                )
                final_content = second_response.choices[0].message.content
        else:
            # If no function call, just get a regular chat response
            final_content = llm_response.content

        full_response = final_content
        message_placeholder.markdown(full_response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})

```
**To run the full application:**
1.  Run the model creation script: `python 1_create_model.py`
2.  Run the FastAPI server: `uvicorn 2_model_api:app --reload`
3.  In a **new terminal**, run the Streamlit app: `streamlit run 3_chatbot_app.py`

### 🧩 Related Concepts

*   **Function Calling / Tool Use:** A capability of modern LLMs where they can be given a set of "tools" (functions described in code) and can decide when to use them based on the conversation. This is the core mechanism enabling this integration.
*   **Agents:** A design pattern where an LLM acts as a reasoning engine to decide a sequence of actions. The LLM can use various tools (like our custom ML API, a web search, or a calculator) to accomplish a complex task. Frameworks like LangChain provide powerful abstractions for building agents.
*   **Microservices Architecture:** The practice of structuring an application as a collection of loosely coupled services. Deploying the ML model as its own microservice (the FastAPI app) is a prime example and a software engineering best practice.
*   **MLOps (Machine Learning Operations):** The discipline of deploying and maintaining ML models in production reliably and efficiently. Having a separate API for the model is the first step in a good MLOps workflow.
*   **Retrieval-Augmented Generation (RAG):** A technique where an LLM's response is grounded in information retrieved from an external knowledge base. In our case, the "knowledge base" is dynamic—it's the output of our custom predictive model.

### 📝 Assignments / Practice Questions

1.  **Multiple Choice Question:**
    In the described architecture, what is the primary role of `st.session_state`?
    A) To store the trained Machine Learning model.
    B) To manage the user's API keys securely.
    C) To persist the conversation history across user interactions and app reruns.
    D) To define the API endpoints for the ML model.

2.  **Short Question:**
    Why is it generally a bad idea to load and run the `model.predict()` function directly inside the Streamlit script instead of calling it via a separate API? Name at least two reasons.

3.  **System Design Problem:**
    You are tasked with building a "Plant Disease Detector" chatbot. A user will describe the symptoms of their plant (e.g., "the leaves have yellow spots and are wilting").
    *   You have a custom image classification model that can identify diseases from a plant photo.
    *   You also have a custom text classification model that can predict diseases from a textual description.
    Design the workflow for a Streamlit app where the LLM can decide which of the two custom models to use based on the user's input (i.e., whether they provide text or upload an image).

4.  **Coding Task:**
    Modify the `2_model_api.py` (FastAPI) script to handle batch predictions. The `/predict` endpoint should be able to accept a list of `InputData` objects and return a list of corresponding predictions.

5.  **Case Study Analysis:**
    A company's Streamlit chatbot for predicting house prices is slow. Users report that the app freezes for several seconds after they submit their query. Upon investigation, you find that the custom house price prediction model (a large deep learning model) is being loaded from disk every time a user sends a message. How would you refactor the architecture using FastAPI and Streamlit to solve this performance issue?

### 📈 Applications

*   **Interactive Data Analysis:** A business analyst can have a conversation with a chatbot to get sales forecasts. The LLM translates "What are the projected sales for next quarter in Germany?" into an API call to a custom forecasting model.
*   **Healthcare Triage:** A patient can describe their symptoms to a chatbot. The LLM can extract key information and query a custom ML model to suggest whether the situation is 'non-urgent', 'requires a doctor's visit', or is an 'emergency'.
*   **Personalized E-commerce:** A user tells a chatbot, "I'm looking for running shoes for a marathon." The LLM uses this context to call a custom recommendation engine that predicts the best shoes based on the user's purchase history and the specific request.
*   **Automated Customer Support:** A customer complains, "My internet is down again!" The LLM extracts the customer ID and calls a custom ML model that predicts the probability of a network outage in their specific area, allowing the bot to give an informed, immediate response.

### 🔗 Related Study Resources

*   **Streamlit Documentation:** Official guides on creating chat elements and using session state.
    *   **Link:** [https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps](https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps)
*   **FastAPI Documentation:** The best place to learn how to build high-performance APIs for your models.
    *   **Link:** [https://fastapi.tiangolo.com/](https://fastapi.tiangolo.com/)
*   **LangChain Documentation:** For advanced use cases involving agents and chains of LLM calls.
    *   **Link:** [https://python.langchain.com/docs/get_started/introduction](https://python.langchain.com/docs/get_started/introduction)
*   **OpenAI Function Calling:** The official documentation on how to use the function calling feature.
    *   **Link:** [https://platform.openai.com/docs/guides/function-calling](https://platform.openai.com/docs/guides/function-calling)
*   **Deploying ML Models with Docker:** A great next step is to containerize your FastAPI app for robust deployment.
    *   **Link:** [https://testdriven.io/blog/dockerizing-fastapi-and-react/](https://testdriven.io/blog/dockerizing-fastapi-and-react/) (While it mentions React, the FastAPI/Docker part is universal).

### 🎯 Summary / Key Takeaways

*   **Decoupled Architecture is Key:** Separate your ML model (API), orchestration logic (LLM), and front-end (Streamlit) for a scalable, maintainable, and efficient application.
*   **LLM as the "Natural Language" Bridge:** The LLM's primary role is to act as an intelligent orchestrator, translating unstructured user language into structured API calls for your specialized model.
*   **Custom Models Provide Precision:** Ground your chatbot's responses in the accurate, data-driven predictions of your custom ML model to avoid hallucinations and provide real value.
*   **Streamlit for Rapid UI Development:** Use Streamlit's chat elements and session state to quickly build a professional-looking and functional user interface without needing front-end development experience.
*   **FastAPI for Robust Model Serving:** Expose your ML model via a FastAPI endpoint to ensure high performance, automatic data validation, and clear documentation.
*   **Start Simple:** The "Chatbot → LLM → API → Model" pattern is a powerful starting point for creating sophisticated AI-powered applications.
