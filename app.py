import streamlit as st
import tensorflow as tf
import numpy as np

# ----------------------------
# Page Setup
# ----------------------------
st.set_page_config(page_title="RadioML Automatic Modulation Classifier")

st.title("📡 RadioML Automatic Modulation Classifier")

# ----------------------------
# Load Model
# ----------------------------
model = tf.keras.models.load_model("radioml_week4_final_model.keras")

st.success("✅ Model loaded successfully!")

# ----------------------------
# Upload IQ Signal
# ----------------------------
uploaded_file = st.file_uploader(
    "Upload a .npy IQ signal",
    type=["npy"]
)

# ----------------------------
# Prediction
# ----------------------------
if uploaded_file is not None:

    signal = np.load(uploaded_file)

    st.write("Original shape:", signal.shape)

    # Convert uploaded signal into the shape expected by the CNN
    if signal.shape == (2, 128):
        signal = signal.reshape(1, 2, 128, 1)

    elif signal.shape == (2, 128, 1):
        signal = np.expand_dims(signal, axis=0)

    elif signal.shape == (1, 2, 128, 1):
        pass

    else:
        st.error(f"Unsupported input shape: {signal.shape}")
        st.stop()

    st.write("Model input shape:", signal.shape)

    # Predict
    prediction = model.predict(signal, verbose=0)

    predicted_class = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    st.subheader("Prediction")

    st.write(f"**Predicted Class Index:** {predicted_class}")
    st.write(f"**Confidence:** {confidence*100:.2f}%")

    st.write("Raw Output Probabilities:")
    st.write(prediction)