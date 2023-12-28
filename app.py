import streamlit as st
import pandas as pd
import tensorflow as tf
import cv2
import numpy as np
from keras.preprocessing import image
# Load your pre-trained model
model_path = 'final_food_classifier.h5'
model = tf.keras.models.load_model(model_path)

# Reading Class Names
df=pd.read_csv('class_names.csv')
label=df['Label']
# Streamlit app
st.title("Food Classification App")

# Upload image through Streamlit
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg"])

# Make prediction on the uploaded image
if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)

    # Read the uploaded image using OpenCV
    img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
    img = cv2.resize(img, (224, 224))  # Resize image to match the model's expected input size
    def preprocess_image(img):
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    img=preprocess_image(img)
    predictions=model.predict(img)
    # Display the top prediction
    st.subheader("Prediction:")
    st.write(label[np.argmax(predictions)])
