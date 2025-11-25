import streamlit as st
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
import os

st.set_page_config(page_title="Azure Key Phrase Extractor", page_icon="📝")

st.title("📝 Azure Key Phrase Extraction App")
st.write("Paste your text below and extract important **topics**, **keywords**, and **phrases** using Azure AI Language Services.")

# ----------------------------
# Load Azure Credentials
# ----------------------------
try:
    endpoint = os.environ["LANGUAGE_ENDPOINT"]
    key = os.environ["LANGUAGE_KEY"]
except KeyError:
    endpoint = ""
    key = ""

client = TextAnalyticsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(key)
)

# ----------------------------
# User Input Box
# ----------------------------
text = st.text_area(
    "✏️ Enter your text here:",
    height=200,
    placeholder="Write or paste a paragraph here to extract key phrases..."
)

# ----------------------------
# Process Button
# ----------------------------
if st.button("Extract Key Phrases"):
    if not text.strip():
        st.error("Please enter some text first.")
    else:
        with st.spinner("Extracting key phrases using Azure AI..."):
            response = client.extract_key_phrases(documents=[text])
            result = response[0]

        st.success("Extraction complete!")

        # ----------------------------
        # Display Results
        # ----------------------------
        if not result.is_error:
            st.subheader("📌 Key Phrases Found:")
            
            if len(result.key_phrases) == 0:
                st.info("No key phrases detected.")
            else:
                for phrase in result.key_phrases:
                    st.write(f"- **{phrase}**")

        else:
            st.error(f"An error occurred: {result.error.message}")
