## Integrating LLMs with Custom ML Models using Streamlit: A Step-by-Step Tutorial

### 📘 Introduction

In the age of artificial intelligence, conversational interfaces are becoming the new frontier for user interaction. Large Language Models (LLMs) like those accessible through APIs from Groq, OpenAI, or Hugging Face have made it incredibly easy to build chatbots that can understand and respond to human language. However, the true power of these models is unlocked when they are combined with specialized, custom-trained machine learning (ML) models.

This tutorial will guide you through the process of integrating a powerful LLM with a custom ML model to create an intelligent, interactive application using Streamlit. We will build a **Heart Disease Prediction chatbot**. A user will be able to describe their clinical parameters in natural language (e.g., "I am 55 years old with a cholesterol level of 220 mg/dL"). The LLM will parse this unstructured text into a structured format that our custom ML model can understand. The ML model will then predict the likelihood of heart disease, and the result will be presented back to the user in a friendly, conversational manner.

**Why does this matter?**

*   **Enhanced User Experience:** Instead of filling out rigid forms, users can interact with your application naturally, as if they were talking to a human expert.
*   **Increased Accessibility:** This approach makes complex ML models accessible to non-technical users who may not understand the specific input features required.
*   **Dynamic and Smart Applications:** The LLM can not only parse data but also provide explanations, answer follow-up questions, and guide the user, creating a truly intelligent system.

**Scope:** This tutorial will cover everything from loading a pre-trained Keras model to building the Streamlit interface, parsing user input with an LLM, making predictions, and presenting the results.

### 🔍 Deep Explanation

The core logic of our application follows a clear, multi-step process. This architecture allows for a clean separation of concerns, where each component has a specific role.

**The Architectural Flow:**

1.  **User Input:** The user interacts with a chat interface built with Streamlit, entering their health data in plain English.
2.  **LLM-Powered Parsing:** The unstructured text from the user is sent to an LLM via an API call. We will engineer a prompt that instructs the LLM to act as a data extractor, identifying key medical parameters and structuring them into a predefined JSON format.
3.  **Structured Data Conversion:** The LLM returns a JSON object (which we will handle as a Python dictionary) containing the parsed data (e.g., `{'age': 55, 'chol': 220, ...}`).
4.  **Data Preprocessing:** The structured data is then prepared for the ML model. This step often involves scaling numerical features to match the format used during the model's training phase.
5.  **ML Model Prediction:** The preprocessed data is fed into our custom-trained Keras model, which outputs a prediction (e.g., a probability score for the presence of heart disease).
6.  **LLM-Powered Explanation:** The raw prediction (e.g., `0.85`) is sent back to the LLM. We use another prompt to ask the LLM to translate this technical output into a user-friendly, empathetic explanation.
7.  **Displaying the Result:** The final, human-readable response is displayed to the user in the Streamlit chat window.

Here is a diagram illustrating this flow:

```
User Input (Natural Language)
       |
       v
Streamlit Chat Interface
       |
       v
LLM API (Data Parsing) --> Returns Structured JSON
       |
       v
Python Backend (Preprocessing)
       |
       v
Custom ML Model (.h5) --> Returns Prediction (e.g., 0 or 1)
       |
       v
LLM API (Explanation Generation) --> Returns Friendly Text
       |
       v
Streamlit Chat Interface
       |
       v
User (Receives Explained Prediction)
```

### 💡 Prerequisites

Before we begin, ensure you have the following:

*   **Python 3.8+** installed on your system.
*   **Basic understanding of Python**, including dictionaries, lists, and functions.
*   **An API key from an LLM provider.** For this tutorial, we will use the **Groq API**, which is known for its high speed and offers a generous free tier.
    *   Go to [GroqCloud](https://console.groq.com/keys) to sign up and get your free API key.
*   **Required Python libraries installed.** You can install them all with the following command:
    ```bash
    pip install streamlit tensorflow scikit-learn groq pandas numpy
    ```

### Step 1: Create and Load the Custom ML Model

For this tutorial, we'll use the well-known **UCI Heart Disease Dataset**. First, we need a trained ML model. Below is a Python script (`train_model.py`) that trains a simple neural network using Keras (TensorFlow) and saves it as `heart_disease_model.h5`. It also saves the scaler used for preprocessing, which is crucial for making accurate predictions on new data.

**Dataset:** You can download the dataset from Kaggle: [Heart Disease UCI](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data). Save it as `heart_disease.csv` in your project directory.

**`train_model.py`**
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib

# 1. Load the dataset
data = pd.read_csv('heart.csv')

# 2. Define features (X) and target (y)
X = data.drop('target', axis=1)
y = data['target']

# 3. Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Build the Keras model
model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

# 6. Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
# 7. Train the model
model.fit(X_train_scaled, y_train, epochs=100, batch_size=10, verbose=1)

# 8. Save the model and the scaler
model.save('heart_disease_model.h5')
joblib.dump(scaler, 'scaler.joblib')

print("Model and scaler have been saved successfully!")

# Evaluate the model (optional)
loss, accuracy = model.evaluate(X_test_scaled, y_test)
print(f"Test Accuracy: {accuracy*100:.2f}%")
```

Run this script once to generate `heart_disease_model.h5` and `scaler.joblib`. Now, our Streamlit app can load these files to make predictions.

### Step 2: Set Up the Streamlit Application

Create a new file named `streamlit_app.py`. This will be the main file for our application. We'll start by setting up the basic Streamlit interface, including the title and the chat history.

**`streamlit_app.py` (Initial Setup)**
```python
import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import joblib
from groq import Groq
import json
import pandas as pd

# --- Page Configuration ---
st.set_page_config(
    page_title="Heart Disease Prediction Chatbot",
    page_icon="❤️",
    layout="centered"
)

# --- Title and Description ---
st.title("❤️ Heart Disease Prediction Chatbot")
st.markdown("""
Welcome! I'm a chatbot powered by an AI model that can help you understand your risk of heart disease.
Please describe your medical parameters in the chat below. For example:
*_"I am a 52 year old male with a resting blood pressure of 130, cholesterol of 240, and a max heart rate of 150."_*
""")

# --- Load ML Model and Scaler ---
@st.cache_resource
def load_prediction_model():
    model = load_model('heart_disease_model.h5')
    scaler = joblib.load('scaler.joblib')
    # Feature names based on the training data columns
    feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    return model, scaler, feature_names

model, scaler, feature_names = load_prediction_model()

# --- Initialize Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you today?"}]

# --- Display Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Main App Logic (to be continued) ---

```

### Step 3: Collect and Parse User Input with an LLM

This is the most critical step. We need to take the user's free-text input and convert it into a structured dictionary. We will use the Groq API for this. We will craft a system prompt that tells the LLM exactly what to do: extract specific medical features and return them as a JSON object.

First, add a section in your `streamlit_app.py` to get the Groq API key from the user (using `st.sidebar` and `st.text_input` for security).

**`streamlit_app.py` (Adding API Key Input)**
```python
# ... (previous code)

# --- Groq API Key ---
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except:
    groq_api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

if not groq_api_key:
    st.info("Please add your Groq API key to continue.")
    st.stop()

client = Groq(api_key=groq_api_key)

# ... (rest of the code)
```
*Note: For deployment, it's best practice to use Streamlit's secrets management (`st.secrets`). For local development, the sidebar input is fine.*

Now, let's create the function that calls the Groq API to parse the input.

**`streamlit_app.py` (Adding the LLM Parser Function)**
```python
# ... (previous code)

def get_structured_data(user_input):
    """
    Uses Groq LLM to parse user input into a structured dictionary.
    """
    system_prompt = f"""
    You are an expert data extraction assistant. Your task is to extract medical parameters
    from the user's text and return them as a JSON object. The required features are:
    {', '.join(feature_names)}.

    - The user will provide their data in natural language.
    - You must map their input to the correct feature names.
    - For binary features ('sex', 'fbs', 'exang'), use 1 for male/true/yes and 0 for female/false/no.
    - If a feature is not mentioned, set its value to 0.
    - Return ONLY the JSON object, with no other text or explanations.

    Example:
    User input: "I am a 55 year old man, BP 140/90, cholesterol 250, my fasting blood sugar is fine, and I don't get angina on exercise. Max heart rate 160."
    Your output:
    {{
      "age": 55, "sex": 1, "cp": 0, "trestbps": 140, "chol": 250,
      "fbs": 0, "restecg": 0, "thalach": 160, "exang": 0, "oldpeak": 0.0,
      "slope": 0, "ca": 0, "thal": 0
    }}
    """
    
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ],
        model="llama-3.1-8b-instant",
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    
    response_content = chat_completion.choices[0].message.content
    try:
        return json.loads(response_content)
    except json.JSONDecodeError:
        st.error("Error: The LLM returned an invalid format. Please try rephrasing your input.")
        return None

# ... (rest of the code)
```

### Step 4 & 5: Preprocess, Predict, and Explain

With the structured data in hand, we can now preprocess it, feed it to our Keras model, and get a prediction. We will then send this prediction *back* to the LLM to generate a human-friendly explanation.

**`streamlit_app.py` (Adding Prediction and Explanation Logic)**
```python
# ... (previous code)

def get_prediction_explanation(prediction_result):
    """
    Uses Groq LLM to generate a user-friendly explanation of the prediction.
    """
    if prediction_result == 1:
        prompt_text = "The model predicts a HIGH risk of heart disease. Please provide a brief, empathetic explanation for a non-technical user. Advise them to consult a doctor for a proper diagnosis. Do not give medical advice."
    else:
        prompt_text = "The model predicts a LOW risk of heart disease. Please provide a brief, reassuring explanation for a non-technical user. Encourage them to maintain a healthy lifestyle."

    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful medical assistant AI. Your role is to explain prediction results clearly and kindly."},
            {"role": "user", "content": prompt_text},
        ],
        model="llama-3.1-8b-instant",
        temperature=0.5,
    )
    return chat_completion.choices[0].message.content

# --- Main App Logic ---
if prompt := st.chat_input("Describe your health parameters..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in a stream-like fashion
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # Step 1: Parse user input
        message_placeholder.markdown("Parsing your input... ⚙️")
        structured_data = get_structured_data(prompt)
        
        if structured_data:
            message_placeholder.markdown("Analyzing your data with the ML model... 🧠")
            
            # Step 2: Prepare data for the model
            try:
                # Create a DataFrame to ensure feature order
                input_df = pd.DataFrame([structured_data])
                input_df = input_df[feature_names] # Ensure correct column order

                # Step 3: Scale the features
                input_scaled = scaler.transform(input_df)

                # Step 4: Make a prediction
                prediction_prob = model.predict(input_scaled)[0][0]
                prediction = 1 if prediction_prob > 0.5 else 0

                # Step 5: Get a friendly explanation
                message_placeholder.markdown("Generating an explanation... ✍️")
                explanation = get_prediction_explanation(prediction)
                
                # Display the final result
                result_md = f"""
                ### Prediction Result:
                - **Risk Level:** {'High Risk' if prediction == 1 else 'Low Risk'}
                - **Probability Score:** {prediction_prob:.2f}

                ---
                **Explanation:**
                {explanation}
                """
                message_placeholder.markdown(result_md)
                st.session_state.messages.append({"role": "assistant", "content": result_md})

            except Exception as e:
                error_message = f"An error occurred during prediction: {e}. Please ensure your input is clear and contains relevant medical terms."
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
        else:
            error_msg = "I couldn't understand the medical data in your message. Could you please try rephrasing it?"
            message_placeholder.markdown(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
```

### Step 6: Full Example and Running the App

Here is the complete `streamlit_app.py` file. Save it in the same directory as `heart_disease_model.h5`, `scaler.joblib`, and `heart_disease.csv`.

**`streamlit_app.py`**
```python
import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import joblib
from groq import Groq
import json
import pandas as pd

# --- Page Configuration ---
st.set_page_config(
    page_title="Heart Disease Prediction Chatbot",
    page_icon="❤️",
    layout="centered"
)

# --- Title and Description ---
st.title("❤️ Heart Disease Prediction Chatbot")
st.markdown("""
Welcome! I'm a chatbot powered by an AI model that can help you understand your risk of heart disease.
Please describe your medical parameters in the chat below. For example:
*_"I am a 52 year old male with a resting blood pressure of 130, cholesterol of 240, and a max heart rate of 150."_*
""")

# --- Groq API Key Handling ---
try:
    # For Streamlit Community Cloud
    groq_api_key = st.secrets["GROQ_API_KEY"]
except (KeyError, FileNotFoundError):
    # For local development
    groq_api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

if not groq_api_key:
    st.info("Please add your Groq API key to continue.")
    st.stop()

client = Groq(api_key=groq_api_key)

# --- Load ML Model and Scaler ---
@st.cache_resource
def load_prediction_model():
    """Loads the pre-trained Keras model and the scaler."""
    try:
        model = load_model('heart_disease_model.h5')
        scaler = joblib.load('scaler.joblib')
        # These are the feature names the model was trained on, in order.
        feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        return model, scaler, feature_names
    except Exception as e:
        st.error(f"Error loading model or scaler: {e}")
        st.stop()

model, scaler, feature_names = load_prediction_model()

# --- Initialize Session State for Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you today?"}]

# --- Display Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- LLM and Prediction Functions ---
def get_structured_data(user_input):
    """Uses Groq LLM to parse user input into a structured dictionary."""
    system_prompt = f"""
    You are an expert data extraction assistant. Your task is to extract medical parameters
    from the user's text and return them as a JSON object. The required features are:
    {', '.join(feature_names)}.

    - The user will provide their data in natural language.
    - You must map their input to the correct feature names.
    - For binary features ('sex', 'fbs', 'exang'), use 1 for male/true/yes and 0 for female/false/no.
    - If a value for a feature is not mentioned, you MUST set its value to a reasonable default, typically 0.
    - Your output MUST be only the JSON object, with no other text, explanations, or markdown formatting.
    """
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ],
            model="llama3-8b-8192",
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        response_content = chat_completion.choices[0].message.content
        return json.loads(response_content)
    except Exception as e:
        st.error(f"An error occurred with the LLM API: {e}")
        return None

def get_prediction_explanation(prediction_result):
    """Uses Groq LLM to generate a user-friendly explanation of the prediction."""
    if prediction_result == 1:
        prompt_text = "The model predicts a HIGH risk of heart disease. Please provide a brief, empathetic explanation for a non-technical user. IMPORTANT: Advise them to consult a doctor for a proper diagnosis and that this is not a medical opinion."
    else:
        prompt_text = "The model predicts a LOW risk of heart disease. Please provide a brief, reassuring explanation for a non-technical user and encourage them to maintain a healthy lifestyle."

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful medical assistant AI. Your role is to explain prediction results clearly and kindly."},
                {"role": "user", "content": prompt_text},
            ],
            model="llama3-8b-8192",
            temperature=0.5,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Could not generate explanation due to an API error: {e}"


# --- Main Application Logic ---
if prompt := st.chat_input("Describe your health parameters..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # Step 1: Parse user input
        message_placeholder.markdown("Parsing your input... ⚙️")
        structured_data = get_structured_data(prompt)
        
        if structured_data:
            message_placeholder.markdown("Analyzing your data with the ML model... 🧠")
            
            try:
                # Step 2: Prepare data for the model
                input_df = pd.DataFrame([structured_data])
                input_df = input_df[feature_names] # Ensure correct feature order

                # Step 3: Scale the features
                input_scaled = scaler.transform(input_df)

                # Step 4: Make a prediction
                prediction_prob = model.predict(input_scaled)[0][0]
                prediction = 1 if prediction_prob > 0.5 else 0

                # Step 5: Get a friendly explanation
                message_placeholder.markdown("Generating an explanation... ✍️")
                explanation = get_prediction_explanation(prediction)
                
                result_md = f"""
                ### Prediction Result:
                - **Risk Level:** {'**High Risk**' if prediction == 1 else '**Low Risk**'}
                - **Confidence Score:** {prediction_prob:.2f}

                ---
                #### Explanation:
                {explanation}
                
                **Disclaimer:** This is an AI-generated prediction and not a substitute for professional medical advice.
                """
                message_placeholder.markdown(result_md)
                st.session_state.messages.append({"role": "assistant", "content": result_md})

            except Exception as e:
                error_message = f"An error occurred during prediction: {e}. Please ensure your input is clear."
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
        else:
            error_msg = "I couldn't understand the medical data in your message. Could you please try rephrasing it?"
            message_placeholder.markdown(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

```

**To run the application:**

1.  Open your terminal.
2.  Navigate to the directory where you saved the files.
3.  Run the command: `streamlit run streamlit_app.py`
4.  Your browser will open with the chatbot interface.

### 🧩 Related Concepts

*   **Prompt Engineering:** The art of designing effective prompts for LLMs is crucial. In our app, we used two distinct prompts: one for structured data extraction and another for natural language explanation.
*   **Function Calling / Tool Use:** More advanced LLMs offer "function calling" or "tool use" capabilities, which provide a more robust way to get structured JSON output compared to prompt formatting alone.
*   **Model Fine-Tuning:** For highly specialized domains, you might fine-tune a smaller LLM on examples of text-to-JSON conversion to improve accuracy and reduce reliance on complex prompts.
*   **Vector Databases & RAG:** For applications that need to answer questions based on large documents (like medical journals), you would combine this pattern with Retrieval-Augmented Generation (RAG) and a vector database.
*   **MLOps (Machine Learning Operations):** In a production environment, you would need a robust system for versioning your ML models, monitoring their performance, and retraining them as new data becomes available.

### 📝 Assignments / Practice Questions

1.  **Multiple Choice Question:** What is the primary role of the LLM in the initial step of the application's workflow?
    a) To train the machine learning model.
    b) To convert unstructured user text into a structured JSON format.
    c) To create the Streamlit user interface.
    d) To directly predict the user's health risk.

2.  **Short Answer:** Why is it necessary to save and reuse the `StandardScaler` object (`scaler.joblib`) from the training script?

3.  **Problem-Solving Task:** Modify the `get_structured_data` function to handle a new feature, `height` (in cm). You will need to update the `system_prompt` to include this new feature. How would you ensure the Keras model could use this feature?

4.  **Case Study:** Imagine you want to adapt this application for a "Loan Approval Predictor." Your custom ML model requires features like `income`, `credit_score`, `loan_amount`, and `employment_duration`.
    *   What would your new `system_prompt` for the LLM look like?
    *   What kind of user input would you expect?
    *   What challenges might you face in parsing financial data compared to medical data?

5.  **Coding Challenge:** Add error handling to the `get_structured_data` function. If the LLM fails to return a valid JSON object or misses key features, the app should gracefully inform the user and ask them to rephrase their input, rather than crashing.

### 📈 Applications

This powerful pattern of combining conversational AI with specialized predictive models can be applied across numerous industries:

*   **Healthcare:** As demonstrated, for preliminary risk assessment of diseases like diabetes, cancer, or stroke.
*   **Finance:** For loan approval chatbots, fraud detection systems, and personalized investment advice tools.
*   **Customer Support:** Automating complex troubleshooting by having an LLM understand the user's problem and feed structured data into a diagnostic model.
*   **E-commerce:** Creating personal shopper bots that understand user preferences (e.g., "I'm looking for a blue cotton shirt for summer") and use a recommendation model to find the perfect product.
*   **Real Estate:** Building a chatbot that helps users find a house by understanding their needs ("I want a 3-bedroom house near a good school") and feeding those parameters into a price prediction or matching model.

### 🔗 Related Study Resources

*   **Streamlit Documentation:** For building chat applications.
*   **Groq API Documentation:** To understand the API parameters and models available.
*   **TensorFlow/Keras Documentation:** For saving and loading models.
*   **Scikit-learn Documentation:** For data preprocessing tools like `StandardScaler`.
*   **GitHub Repository with Code for a similar project:** [Heart Disease Prediction using Machine Learning](https://github.com/g-shreekant/Heart-Disease-Prediction-using-Machine-Learning).

### 🎯 Summary / Key Takeaways

*   **Hybrid AI is Powerful:** Combining the natural language understanding of LLMs with the predictive accuracy of specialized ML models creates highly effective and user-friendly applications.
*   **The Core Pattern:** The fundamental workflow is **User Input -> LLM Parsing -> Structured Data -> ML Prediction -> LLM Explanation -> User Output**. This pattern is generalizable to almost any domain.
*   **Prompt Engineering is Key:** The success of the data parsing and explanation steps depends heavily on well-crafted prompts that clearly define the LLM's task and expected output format.
*   **Streamlit Simplifies UI:** Streamlit is an excellent tool for rapidly prototyping and deploying interactive AI and data science applications with minimal boilerplate code.
*   **Always Preprocess:** Never forget to apply the *exact same* preprocessing (e.g., scaling) to new data as was used to train the model. Failure to do so will result in incorrect predictions.
*   **Safety and Ethics:** When dealing with sensitive data like medical or financial information, always include disclaimers and prioritize user privacy. Never present AI predictions as infallible facts.
