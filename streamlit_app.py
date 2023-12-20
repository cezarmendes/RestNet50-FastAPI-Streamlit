# Streamlit app
import streamlit as st
import requests
from PIL import Image


# Streamlit
st.set_page_config(page_title="Image Classifier with ResNet50", layout="wide")

st.title("Image Classifier using ResNet50")

# Upload widget for image
uploaded_file = st.file_uploader("Choose an image, please!", type=["jpg", "jpeg", "png"])

# API endpoint
API_URL = "http://localhost:8000/classify"

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image')

    st.write("**Classifying...**")

    # Prepare the image for the API
    files = {"image_file": uploaded_file.getvalue()}

    # Call the API
    response = requests.post(API_URL, files=files)
    if response.status_code == 200:

        # Display results
        predictions = response.json()["predictions"]
        st.write("**Predictions:**")
        for i, pred in enumerate(predictions):
            st.write(f"{i+1}. {pred['label']} (Probability: {pred['probability']:.4f})")

    else:
        st.error("Failed to get response from the API.")

# Copy and past in terminal (testing)
# streamlit run streamlit_app.py