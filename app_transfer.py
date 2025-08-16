# app_transfer.py
import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

st.set_page_config(page_title="Brain Tumor Detection (Transfer Learning)", page_icon="🧠")

@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model(
        "models/brain_tumor_mobilenetv2.keras",
        custom_objects={"preprocess_input": preprocess_input}
    )
    with open("models/class_names.json", "r", encoding="utf-8") as f:
        class_names = json.load(f)
    return model, class_names

model, class_names = load_artifacts()

st.title("🧠 Brain Tumor Detection (MobileNetV2)")
st.write("Upload an MRI image (glioma / meningioma / pituitary / no_tumor or notumor).")

uploaded_file = st.file_uploader("Choose an MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    img = img.resize((224, 224))
    x = np.array(img, dtype=np.float32)
    # MobileNetV2 expects preprocess_input ([-1,1] scaling with channel-wise ops)
    x = preprocess_input(x)
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)
    idx = int(np.argmax(preds))
    prob = float(np.max(preds))

    st.success(f"Prediction: **{class_names[idx]}**")
    st.progress(min(1.0, prob))
    st.caption(f"Confidence: {prob*100:.2f}%")
