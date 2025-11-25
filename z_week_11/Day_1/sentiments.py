import streamlit as st
import os
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

st.set_page_config(page_title="Azure Sentiment Analysis App", page_icon="💬")

st.title("💬 Azure Sentiment Analysis with Opinion Mining")
st.write("Enter one or more sentences below and analyze their sentiment using Azure AI Language Services.")

# ----------------------------
# Load Azure Credentials
# ----------------------------
try:
    endpoint = os.environ["LANGUAGE_ENDPOINT"]
    key = os.environ["LANGUAGE_KEY"]
except KeyError:
    endpoint = ""
    key = ""

# Create Azure client
client = TextAnalyticsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(key)
)

# ----------------------------
# User Input
# ----------------------------
user_input = st.text_area(
    "✏️ Enter sentences (one per line):",
    placeholder="Example:\nThe product is amazing!\nThe delivery was slow.\nEl servicio al cliente fue excelente."
)

if st.button("Analyze Sentiment"):
    if not user_input.strip():
        st.error("Please enter at least one sentence.")
    else:
        reviews = [line.strip() for line in user_input.split("\n") if line.strip()]

        with st.spinner("Analyzing with Azure AI..."):
            response = client.analyze_sentiment(
                documents=reviews,
                show_opinion_mining=True
            )
            results = [doc for doc in response if not doc.is_error]

        st.success("Analysis completed!")

        # ----------------------------
        # Display Results
        # ----------------------------
        for idx, doc in enumerate(results):
            review_text = reviews[idx]

            st.subheader(f"📝 Review {idx+1}")
            st.write(f"**Text:** {review_text}")

            st.write(f"**Overall Sentiment:** `{doc.sentiment.upper()}`")
            st.write(
                f"**Scores** → Positive: `{doc.confidence_scores.positive:.2f}`, "
                f"Negative: `{doc.confidence_scores.negative:.2f}`, "
                f"Neutral: `{doc.confidence_scores.neutral:.2f}`"
            )

            # Sentence-level analysis
            with st.expander("🔍 Sentence-Level Opinion Mining"):
                for sentence in doc.sentences:
                    st.write(f"**Sentence:** {sentence.text}")
                    st.write(f"- Sentiment: `{sentence.sentiment}`")

                    # Opinion mining details
                    for mined_opinion in sentence.mined_opinions:
                        target = mined_opinion.target
                        assessments = [a.text for a in mined_opinion.assessments]

                        st.write(f"  - **Target:** {target.text}")
                        st.write(f"  - **Assessments:** {assessments}")

            st.markdown("---")
