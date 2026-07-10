import streamlit as st
import tensorflow as tf
import numpy as np

st.set_page_config(page_title="RadioML Classifier")

st.title("📡 RadioML Automatic Modulation Classifier")

model = tf.keras.models.load_model("radioml_week4_final_model.keras")

st.success("Model loaded successfully!")

uploaded_file = st.file_uploader(
    "Upload a .npy IQ signal",
    type=["npy"]
)

if uploaded_file is not None:

    signal = np.load(uploaded_file)

    st.write("Original shape:", signal.shape)

    if signal.shape == (2,128):
        signal = signal.reshape(1,2,128,1)

    prediction = model.predict(signal)

    predicted_class = np.argmax(prediction)

    confidence = np.max(prediction)

    st.write("Predicted class:", predicted_class)

    st.write(f"Confidence: {confidence*100:.2f}%")