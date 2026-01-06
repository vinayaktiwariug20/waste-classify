# app_cloud.py
import json
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import streamlit as st
import tensorflow as tf
import os

# Get the directory where app_batch.py is located
BASE_DIR = os.path.dirname(__file__)

# Use relative paths instead of absolute D:\ paths
MODEL_PATH = os.path.join(BASE_DIR, "EfficientNetB0_best.keras")
META_PATH  = os.path.join(BASE_DIR, "export_fixed", "metadata.json")

# Update your caption so it doesn't leak your local D: drive info
st.caption("Model: EfficientNetB0 (Waste Classification)")

# ==== load model + metadata ====
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    with open(META_PATH, "r") as f:
        meta = json.load(f)
    return model, meta

model, meta = load_assets()
IMG_SIZE = tuple(meta.get("img_size", [224, 224]))
CLASS_NAMES = meta.get("classes", [])
st.write("Input shape:", model.input_shape)
st.write("Classes:", ", ".join(CLASS_NAMES))

# ==== preprocessing helpers ====
def center_square_crop(pil_img: Image.Image) -> Image.Image:
    w, h = pil_img.size
    side = min(w, h)
    left = (w - side) // 2
    top  = (h - side) // 2
    return pil_img.crop((left, top, left + side, top + side))

def preprocess_for_model(pil_img: Image.Image):
    """
    Returns:
      arr_resized: np.float32 [H,W,3] in 0..255 for current model
      view_resized: PIL.Image 224x224 (simple resize)
      view_crop:    PIL.Image 224x224 (center-crop then resize)
    """
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    view_resized = pil_img.resize(IMG_SIZE, Image.BILINEAR)
    cropped = center_square_crop(pil_img)
    view_crop = cropped.resize(IMG_SIZE, Image.BILINEAR)

    # IMPORTANT: your current checkpoint expects 0..255
    arr_resized = np.array(view_resized).astype(np.float32)
    return arr_resized, view_resized, view_crop

# UI toggle: which view to feed into the model
use_center_crop = st.checkbox("Use center-crop for model input (instead of simple resize)", value=False)

# ==== uploader ====
uploads = st.file_uploader(
    "Upload one or more images",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)

if uploads:
    batch = []
    originals = []
    views_resized = []
    views_cropped = []
    names = []

    for file in uploads:
        pil = Image.open(file)
        arr_resized, view_resized, view_crop = preprocess_for_model(pil)
        originals.append(pil)
        views_resized.append(view_resized)
        views_cropped.append(view_crop)
        names.append(file.name)

        # choose which tensor to feed
        if use_center_crop:
            arr = np.array(view_crop).astype(np.float32)  # 0..255
        else:
            arr = arr_resized
        batch.append(arr)

    batch = np.stack(batch, axis=0)  # [N,224,224,3]
    st.info(f"Prepared batch of {len(batch)} images with shape {batch.shape}")

    if st.button("Predict batch"):
        with st.spinner("Predicting..."):
            probs = model(batch, training=False).numpy()  # [N,C]
        top3 = np.argsort(-probs, axis=1)[:, :3]

        for i in range(len(batch)):
            st.markdown("---")
            st.subheader(names[i])

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("Original")
                st.image(originals[i], use_container_width=True)
            with col2:
                st.markdown("Model view (resize 224x224)")
                st.image(views_resized[i], use_container_width=True)
            with col3:
                st.markdown("Model view (center-crop → 224x224)")
                st.image(views_cropped[i], use_container_width=True)

            labels = [CLASS_NAMES[idx] for idx in top3[i]]
            vals   = [float(probs[i, idx]) for idx in top3[i]]
            st.success(f"Top class: {labels[0]}  |  conf: {vals[0]:.3f}")

            # top-3 as dict
            st.write({k: round(v, 4) for k, v in zip(labels, vals)})

            # bar chart (needs a DataFrame with named columns)
            df_top = pd.DataFrame({"label": labels, "probability": vals})
            st.bar_chart(df_top, x="label", y="probability")

            # full table sorted
            df_full = pd.DataFrame({
                "class": CLASS_NAMES,
                "probability": probs[i].tolist()
            }).sort_values("probability", ascending=False)
            st.dataframe(df_full, use_container_width=True, hide_index=True)
