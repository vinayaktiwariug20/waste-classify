import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Page configuration
st.set_page_config(page_title="WasteVision AI", page_icon="♻️")

@st.cache_resource
def load_waste_model():
    """Load the pre-trained EfficientNetB0 model."""
    return tf.keras.models.load_model('EfficientNetB0_best.keras')

def process_and_predict(image, model):
    """Pre-process image and return top 3 predictions with confidence levels."""
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    preds = model.predict(img_array)[0]
    classes = ['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash', 'Fruit', 'Vegetable', 'Textile']
    
    # Get top 3 predictions
    top_3_idx = np.argsort(preds)[-3:][::-1]
    top_3_results = [(classes[idx], preds[idx]) for idx in top_3_idx]
    
    return top_3_results

# UI Elements
st.title("♻️ Waste Classification")
st.write("Upload an image to classify it using the EfficientNetB0 backbone.")

file = st.file_uploader("Upload Waste Image", type=["jpg", "png", "jpeg"])

if file:
    img = Image.open(file)
    st.image(img, use_container_width=True)
    
    with st.spinner('Analyzing features...'):
        model = load_waste_model()
        label, confidence = process_and_predict(img, model)
        
    st.success(f"Classification: {label}")
    st.progress(float(confidence))
    st.write(f"Confidence: {confidence:.2%}")

st.divider()
st.caption("Built with TensorFlow & Streamlit | EfficientNet-B0")