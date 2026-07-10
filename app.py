import streamlit as st
import tensorflow as tf

st.title("RadioML Test")

st.write("TensorFlow version:", tf.__version__)

st.write("Before loading model...")

try:
    model = tf.keras.models.load_model("radioml_week4_final_model.keras")
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(e)