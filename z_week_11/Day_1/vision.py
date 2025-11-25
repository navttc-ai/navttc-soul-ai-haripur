import streamlit as st
import tempfile
import os
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential


# -----------------------------
# 🔑 Azure Credentials Handling
# -----------------------------

st.title("🔍 Azure Computer Vision - Image Analyzer")
st.write("Upload an image to get description, objects, and tags.")

# Try to fetch secrets
try:
    endpoint = st.secrets["VISION_ENDPOINT"]
    key = st.secrets["VISION_KEY"]
except Exception:
    # Ask user to input keys manually if not deployed with Streamlit secrets
    st.sidebar.header("🔐 Azure API Settings")
    endpoint = st.sidebar.text_input("Azure Vision Endpoint")
    key = st.sidebar.text_input("Azure Vision API Key", type="password")

if endpoint and key:
    client = ImageAnalysisClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(key)
    )
else:
    st.warning("Please provide Azure Vision API credentials in the sidebar.")
    st.stop()


# -----------------------------
# 📤 Image Upload
# -----------------------------

uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_image:

    # Display uploaded image
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Save the uploaded file temporarily
    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, uploaded_image.name)

    with open(temp_file_path, "wb") as f:
        f.write(uploaded_image.getbuffer())

    st.write("📡 Analyzing image... please wait")

    # -----------------------------
    # 🔍 Azure Image Analysis
    # -----------------------------
    visual_features = [
        VisualFeatures.CAPTION,
        VisualFeatures.OBJECTS,
        VisualFeatures.TAGS,
    ]

    try:
        result = client.analyze(
            image_data=open(temp_file_path, "rb"),
            visual_features=visual_features
        )

        st.subheader("📌 Image Description")
        if result.caption:
            st.write(f"**{result.caption.text}**")
            st.write(f"Confidence: {result.caption.confidence * 100:.2f}%")

        st.subheader("🧩 Detected Objects")
        if result.objects:
            for obj in result.objects.list:
                st.write(
                    f"- **{obj.tags[0].name}** "
                    f"(Confidence: {obj.tags[0].confidence * 100:.1f}%)"
                )
        else:
            st.write("No objects detected.")

        st.subheader("🏷️ Tags")
        if result.tags:
            for tag in result.tags.list:
                st.write(f"- {tag.name} ({tag.confidence * 100:.1f}%)")

    except Exception as e:
        st.error(f"Error analyzing image: {e}")
