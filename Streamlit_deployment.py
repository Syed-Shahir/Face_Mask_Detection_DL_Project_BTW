import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from PIL import Image

# Load the trained model
file_path = 'face_mask_detection_model.keras'
model = load_model(file_path)

# Define class names
class_names = ['Without Mask', 'With Mask']

# Set title for the Streamlit app
st.title("Face Mask Detection")

# Allow the user to upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open the uploaded image using PIL
    image = Image.open(uploaded_file)

    # Display the image using Streamlit
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Convert the image to RGB and resize to (128,128) for model prediction
    img_array = np.array(image.convert('RGB').resize((128, 128)))

    # Normalize the image
    img_array = img_array / 255.0

    # Reshape the image to match the input shape of the model
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction using the model
    input_prediction = model.predict(img_array)

    # Get the predicted label
    input_pred_label = np.argmax(input_prediction)

    # Display the result
    if input_pred_label == 1:
        st.write("Prediction: The person in the image is **wearing a mask**.")
    else:
        st.write("Prediction: The person in the image is **not wearing a mask**.")
