import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras

# Define the input image size (adjust as needed)
image_size = (224, 224)

# Define the base model from TensorFlow Hub
base_model = keras.applications.MobileNetV2(input_shape=image_size + (3,), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze pre-trained layers

# Add custom classification layers
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Load the pre-trained model weights
model.load_weights('D:\DL_Projects\helmet_detection\signal.h5')  # Replace with the path to your saved model weights

st.title("Signal Violation Detection")

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Load and preprocess the uploaded image
    new_image = image.load_img(uploaded_image, target_size=image_size)
    new_image = image.img_to_array(new_image)
    new_image = np.expand_dims(new_image, axis=0)
    new_image = new_image / 255.0  # Rescale the image just like the training data

    # Make predictions using the model
    predictions = model.predict(new_image)

    # You can also threshold the prediction if you want a binary classification result
    binary_prediction = (predictions > 0.5).astype(int)

    if binary_prediction[0] == 1:
        # If the prediction is 1, display the image
        st.image(uploaded_image, caption="Signal Violation Detected", use_column_width=True)
        st.write("The model predicted a signal violation for the image.")
    else:
        st.image(uploaded_image, caption="No Signal Violation Detected", use_column_width=True)
        st.write("The model predicted no signal violation for the image.")
