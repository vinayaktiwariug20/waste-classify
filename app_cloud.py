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

# Show the 9 waste categories
st.subheader("📋 Waste Categories")
categories = ['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash', 'Fruit', 'Vegetable', 'Textile']
cols = st.columns(3)
for idx, category in enumerate(categories):
    with cols[idx % 3]:
        st.write(f"• {category}")

st.divider()

file = st.file_uploader("Upload Waste Image", type=["jpg", "png", "jpeg"])

if file:
    img = Image.open(file)
    st.image(img, use_container_width=True)
    
    with st.spinner('Analyzing features...'):
        model = load_waste_model()
        top_3_results = process_and_predict(img, model)
        
    st.success(f"**Top Classification: {top_3_results[0][0]}**")
    
    st.subheader("🎯 Top 3 Predictions")
    for rank, (label, confidence) in enumerate(top_3_results, 1):
        st.write(f"**{rank}. {label}**")
        st.progress(float(confidence))
        st.write(f"Confidence: {confidence:.2%}")
        if rank < 3:
            st.write("")  # Add spacing between predictions

st.divider()
st.caption("Built with TensorFlow & Streamlit | EfficientNet-B0")