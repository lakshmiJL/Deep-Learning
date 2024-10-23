import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the pre-trained MNIST model
model = tf.keras.models.load_model('mnist_model.h5')

st.title("Handwritten Digit Classifier")

st.write("Upload an image of a handwritten digit (28x28 pixels) for prediction.")

# File uploader for images
uploaded_file = st.file_uploader("Choose an image...", type="png")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    image = image.resize((28, 28))                  # Resize to 28x28 pixels
    image = np.array(image) / 255.0                 # Normalize the image

    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Reshape image for model input
    image = np.expand_dims(image, axis=0)
    
    # Predict
    prediction = model.predict(image)
    predicted_digit = np.argmax(prediction)
    
    st.write(f"Predicted Digit: {predicted_digit}")


#streamlit run app.py