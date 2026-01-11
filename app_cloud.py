import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
from pathlib import Path
import requests
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="WasteVision AI",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        color: #28a745;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #6c757d;
        margin-bottom: 2rem;
    }
    .metric-container {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .prediction-box {
        background-color: #f8f9fa;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .stButton>button {
        background-color: #28a745;
        color: white;
        font-weight: 600;
        border-radius: 5px;
        padding: 0.5rem 2rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #20c997;
    }
</style>
""", unsafe_allow_html=True)

# Load model and metadata
@st.cache_resource
def load_model_and_metadata():
    """Load the trained model and metadata"""
    try:
        model = tf.keras.models.load_model('export_fixed/EfficientNetB0_best.keras')
        
        with open('export_fixed/metadata.json', 'r') as f:
            metadata = json.load(f)
        
        return model, metadata
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Make sure 'export_fixed/EfficientNetB0_best.keras' and 'export_fixed/metadata.json' exist")
        return None, None

def preprocess_image(image, img_size):
    """Preprocess image for model prediction"""
    # Resize image
    image = image.resize(img_size)
    
    # Convert to array and normalize
    img_array = np.array(image)
    
    # Handle grayscale images
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    
    # Handle RGBA images
    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]
    
    # Normalize to [0, 1]
    img_array = img_array.astype(np.float32) / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def load_image_from_url(url):
    """Load image from URL"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        return image.convert('RGB')
    except Exception as e:
        st.error(f"Error loading image from URL: {str(e)}")
        return None

def predict_image(model, image, metadata):
    """Make prediction on preprocessed image"""
    img_size = tuple(metadata['img_size'])
    classes = metadata['classes']
    
    # Preprocess
    processed_img = preprocess_image(image, img_size)
    
    # Predict
    predictions = model.predict(processed_img, verbose=0)[0]
    
    # Get top prediction
    top_idx = np.argmax(predictions)
    top_class = classes[top_idx]
    top_confidence = predictions[top_idx]
    
    # Get all predictions sorted
    sorted_indices = np.argsort(predictions)[::-1]
    sorted_predictions = [(classes[i], predictions[i]) for i in sorted_indices]
    
    return top_class, top_confidence, sorted_predictions

# Main app
def main():
    # Header
    st.markdown('<p class="main-header">♻️ WasteVision AI</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Deep Learning-Powered Waste Classification</p>', unsafe_allow_html=True)
    
    # Load model
    model, metadata = load_model_and_metadata()
    
    if model is None or metadata is None:
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("📋 About")
        st.markdown("""
        **WasteVision AI** classifies waste images into 9 categories using an EfficientNet-B0 backbone.
        
        **Performance:**
        - F1-Score: 0.861
        - AUC: 0.988
        
        **Categories:**
        """)
        for i, cls in enumerate(metadata['classes'], 1):
            st.markdown(f"{i}. {cls.split('-')[1]}")
        
        st.markdown("---")
        st.markdown("**Links:**")
        st.markdown("[🔗 GitHub](https://github.com/vinayaktiwariug20/waste-classify)")
        st.markdown("[🎥 Demo Video](https://www.youtube.com/watch?v=hJXq16BBI3c)")
    
    # Main content
    tab1, tab2 = st.tabs(["📤 Upload Image", "🔗 Image URL"])
    
    with tab1:
        st.markdown("### Upload an image to classify")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'bmp'],
            help="Upload a waste image for classification"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.image(image, caption="Uploaded Image", use_container_width=True)
            
            with col2:
                with st.spinner("🔍 Analyzing image..."):
                    top_class, top_confidence, all_predictions = predict_image(model, image, metadata)
                
                # Display top prediction
                st.markdown("### 🎯 Classification Result")
                st.markdown(f"""
                <div class="prediction-box">
                    <h3 style="color: #28a745; margin: 0;">{top_class.split('-')[1]}</h3>
                    <p style="font-size: 1.5rem; font-weight: 600; margin: 0.5rem 0;">
                        {top_confidence * 100:.1f}% Confidence
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Display all predictions
                st.markdown("### 📊 All Predictions")
                for cls, conf in all_predictions:
                    percentage = float(conf) * 100
                    st.progress(float(conf), text=f"{cls.split('-')[1]}: {percentage:.1f}%")
    
    with tab2:
        st.markdown("### Paste an image URL")
        url = st.text_input(
            "Image URL",
            placeholder="https://example.com/image.jpg",
            help="Enter a direct link to an image file"
        )
        
        if st.button("🔍 Classify from URL"):
            if url:
                image = load_image_from_url(url)
                
                if image is not None:
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.image(image, caption="Image from URL", use_container_width=True)
                    
                    with col2:
                        with st.spinner("🔍 Analyzing image..."):
                            top_class, top_confidence, all_predictions = predict_image(model, image, metadata)
                        
                        # Display top prediction
                        st.markdown("### 🎯 Classification Result")
                        st.markdown(f"""
                        <div class="prediction-box">
                            <h3 style="color: #28a745; margin: 0;">{top_class.split('-')[1]}</h3>
                            <p style="font-size: 1.5rem; font-weight: 600; margin: 0.5rem 0;">
                                {top_confidence * 100:.1f}% Confidence
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Display all predictions
                        st.markdown("### 📊 All Predictions")
                        for cls, conf in all_predictions:
                            percentage = float(conf) * 100
                            st.progress(float(conf), text=f"{cls.split('-')[1]}: {percentage:.1f}%")
            else:
                st.warning("Please enter a valid URL")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6c757d;">
        <p>Built with Streamlit & TensorFlow | Trained on RealWaste Dataset</p>
        <p><a href="https://vinayaktiwariug20.github.io/" style="color: #28a745;">← Back to Portfolio</a></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()