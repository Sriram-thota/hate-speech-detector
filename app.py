import streamlit as st
from transformers import pipeline
import pytesseract
from PIL import Image

# ✅ Link to the installed Tesseract OCR engine
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load hate speech detection model
hate_speech_classifier = pipeline("text-classification", model="unitary/toxic-bert")

# UI Title
st.title("🛡️ Hate Speech Detection Using NLP")
st.markdown("Detect hate speech from text or image input using deep learning (BERT).")

# Sidebar for input type
option = st.sidebar.selectbox("Choose Input Type", ("Text", "Image"))

# For Text Input
if option == "Text":
    user_text = st.text_area("Enter text to analyze:", height=200)
    if st.button("Detect Hate Speech"):
        if user_text.strip():
            with st.spinner("Analyzing..."):
                result = hate_speech_classifier(user_text)[0]
                label = result['label']
                score = result['score']
            st.success(f"Prediction: {label} ({score:.2f})")
        else:
            st.warning("Please enter some text.")

# For Image Input
elif option == "Image":
    uploaded_image = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        if st.button("Extract Text and Detect"):
            with st.spinner("Extracting text..."):
                extracted_text = pytesseract.image_to_string(image)
            st.text_area("Extracted Text:", extracted_text, height=200)

            if extracted_text.strip():
                result = hate_speech_classifier(extracted_text)[0]
                label = result['label']
                score = result['score']
                st.success(f"Prediction: {label} ({score:.2f})")
            else:
                st.warning("No text detected in the image.")
