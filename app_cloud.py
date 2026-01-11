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
    
    # Correct alphabetical order from RealWaste folders
    classes = [
        'Cardboard',           # 0
        'Food Organics',       # 1
        'Glass',               # 2
        'Metal',               # 3
        'Miscellaneous Trash', # 4
        'Paper',               # 5
        'Plastic',             # 6
        'Textile Trash',       # 7
        'Vegetation'           # 8
    ]
    
    # Get top 3 predictions
    top_n = 3
    top_n_idx = np.argsort(preds)[-top_n:][::-1]
    top_n_results = [(classes[idx], preds[idx]) for idx in top_n_idx]
    
    return top_n_results, preds, classes, img_array

# UI Elements
st.title("♻️ WasteVision AI")
st.write("Upload an image to classify waste using the EfficientNetB0 model trained on RealWaste dataset.")

# Show the 9 waste categories
st.subheader("📋 Waste Categories")
st.info("This model classifies waste into 9 categories from the RealWaste dataset")

categories = [
    'Cardboard', 'Food Organics', 'Glass', 'Metal', 'Vegetation',
    'Paper', 'Plastic', 'Textile Trash', 'Miscellaneous Trash'
]
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
        top_3_results, preds, classes, img_array = process_and_predict(img, model)
        
    st.success(f"**Top Classification: {top_3_results[0][0]}**")
    
    st.subheader("🎯 Top 3 Predictions")
    for rank, (label, confidence) in enumerate(top_3_results, 1):
        st.write(f"**{rank}. {label}**")
        st.progress(float(confidence))
        st.write(f"Confidence: {confidence:.2%}")
        if rank < 3:
            st.write("")  # Add spacing between predictions

    st.divider()
    st.write("🔍 DEBUG - Raw Model Output:")
    st.write(f"Image array shape: {img_array.shape}")
    st.write(f"Image array min/max: {img_array.min():.3f} / {img_array.max():.3f}")
    st.write(f"Number of classes: {len(preds)}")
    st.write(f"Predictions sum: {preds.sum():.4f} (should be ~1.0)")
    st.write(f"Highest confidence index: {np.argmax(preds)}")
    st.write(f"Class at that index: {classes[np.argmax(preds)]}")
    st.write("All predictions:")
    for idx, (class_name, conf) in enumerate(zip(classes, preds)):
        st.write(f"  [{idx}] {class_name}: {conf:.4f}")

st.divider()
st.caption("Built with TensorFlow & Streamlit | EfficientNet-B0 trained on RealWaste dataset")