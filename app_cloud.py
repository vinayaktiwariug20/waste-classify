# app_cloud.py
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Page Config
st.set_page_config(page_title="WasteVision AI", page_icon="♻️")

@st.cache_resource
def load_model():
    # Replace with your actual model file path
    return tf.keras.models.load_model('models/waste_classifier.h5')

def predict(image, model):
    # Standard preprocessing for EfficientNet
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array)
    classes = ['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash'] # Example classes
    return classes[np.argmax(predictions)], np.max(predictions)

# UI Layout
st.title("♻️ Waste Classification AI")
st.write("Upload a photo of waste to see how the EfficientNet model classifies it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with st.spinner('Analyzing...'):
        model = load_model()
        label, confidence = predict(image, model)
        
    st.success(f"**Prediction:** {label}")
    st.info(f"**Confidence:** {confidence:.2%}")

# Simple Footer
st.divider()
st.caption("Developed by Vinayak Tiwari | EfficientNet-B0 Backbone")