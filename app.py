import os

os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="RadioML Automatic Modulation Classifier",
    page_icon="📡",
    layout="wide"
)

st.title("📡 RadioML Automatic Modulation Classifier")

# -------------------------------
# Class Names
# -------------------------------

CLASS_NAMES = [
    "OOK",
    "4ASK",
    "8ASK",
    "BPSK",
    "QPSK",
    "8PSK",
    "16PSK",
    "32PSK",
    "16APSK",
    "32APSK",
    "64APSK"
]

# -------------------------------
# Load Model
# -------------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("radioml_week4_final_model.keras")

model = load_model()

st.success("✅ Model loaded successfully!")

# -------------------------------
# Upload Signal
# -------------------------------

uploaded_file = st.file_uploader(
    "Upload a .npy IQ Signal",
    type=["npy"]
)

if uploaded_file is not None:

    signal = np.load(uploaded_file)

    st.write("### Signal Information")
    st.write("Original Shape:", signal.shape)

    # Convert to expected model shape
    if signal.shape == (2, 128):
        signal = signal.reshape(1, 2, 128, 1)

    elif signal.shape == (2, 128, 1):
        signal = signal.reshape(1, 2, 128, 1)

    elif signal.shape == (1, 2, 128, 1):
        pass

    else:
        st.error(f"Unsupported input shape: {signal.shape}")
        st.stop()

    st.write("Model Input Shape:", signal.shape)

    # Prediction
    prediction = model.predict(signal, verbose=0)[0]

    predicted_index = int(np.argmax(prediction))
    predicted_label = CLASS_NAMES[predicted_index]
    confidence = float(np.max(prediction))

    st.divider()

    st.header("Prediction")

    st.success(f"**Predicted Modulation:** {predicted_label}")

    st.metric(
        label="Confidence",
        value=f"{confidence*100:.2f}%"
    )

    # -------------------------------
    # Probability Chart
    # -------------------------------

    st.subheader("Prediction Probabilities")

    fig, ax = plt.subplots(figsize=(8,5))

    ax.barh(CLASS_NAMES, prediction)

    ax.set_xlim(0,1)

    ax.set_xlabel("Probability")

    ax.set_title("Model Confidence Across All Classes")

    st.pyplot(fig)