import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import imutils
import easyocr
import matplotlib.pyplot as plt

# Define the input image size (adjust as needed)
image_size = (224, 224)

# Define the base model from TensorFlow Hub for signal violation detection
base_model_signal = keras.applications.MobileNetV2(input_shape=image_size + (3,), include_top=False, weights='imagenet')
base_model_signal.trainable = False  # Freeze pre-trained layers

# Add custom classification layers for signal violation detection
x_signal = GlobalAveragePooling2D()(base_model_signal.output)
x_signal = Dense(1024, activation='relu')(x_signal)
predictions_signal = Dense(1, activation='sigmoid')(x_signal)

model_signal = Model(inputs=base_model_signal.input, outputs=predictions_signal)

# Load the pre-trained model weights for signal violation detection
model_signal.load_weights('D:\DL_Projects\helmet_detection\signal.h5')  # Replace with the path to your saved model weights

st.title("License Plate and Signal Violation Detection App")

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Load and preprocess the uploaded image for signal violation detection
    new_image = image.load_img(uploaded_image, target_size=image_size)
    new_image_signal = image.img_to_array(new_image)
    new_image_signal = np.expand_dims(new_image_signal, axis=0)
    new_image_signal = new_image_signal / 255.0  # Rescale the image just like the training data

    # Make predictions for signal violation using the model
    predictions_signal = model_signal.predict(new_image_signal)

    # You can also threshold the prediction if you want a binary classification result
    binary_prediction_signal = (predictions_signal > 0.5).astype(int)

    if binary_prediction_signal[0] == 1:
        # If the prediction is 1, display the image for signal violation
        st.image(uploaded_image, caption="Signal Violation Detected", use_column_width=True)
        st.write("The model predicted a signal violation for the image.")
    else:
        # License Plate Detection Code
        # Your second code section for license plate detection (executed only if no signal violation is detected)

        gray = cv2.cvtColor(conv_img, cv2.COLOR_BGR2GRAY)

        bfilter = cv2.bilateralFilter(gray, 11, 17, 17)  # Noise reduction
        edged = cv2.Canny(bfilter, 30, 200)  # Edge detection

        keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(keypoints)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        location = None

        for contour in contours:
            approx = cv2.approxPolyDP(contour, 10, True)
            if len(approx) == 4:
                location = approx
                break

        if location is not None:
            mask = np.zeros(gray.shape, np.uint8)
            new_image = cv2.drawContours(mask, [location], 0, 255, -1)
            new_image = cv2.bitwise_and(conv_img, conv_img, mask=mask)

            (x, y) = np.where(mask == 255)
            (x1, y1) = (np.min(x), np.min(y))
            (x2, y2) = (np.max(x), np.max(y))
            cropped_image = gray[x1:x2 + 1, y1:y2 + 1]
            reader = easyocr.Reader(['en'])
            result = reader.readtext(cropped_image)
            st.text(f"Number of text results: {result}")

            text = ''
            for i in range(0, len(result)):
                new_text = result[i][-2]
                text += new_text

            # Determine the position for text outside the rectangle
            text_position = (location[0][0][0], location[1][0][1] - 20)  # Adjust the Y-coordinate as needed

            # Draw the text
            res = cv2.putText(img, text=text, org=text_position, fontFace=font, fontScale=2, color=(0, 0, 0), thickness=4, lineType=cv2.LINE_AA)

            # Draw the rectangle
            res = cv2.rectangle(img, tuple(location[0][0]), tuple(location[2][0]), (0, 255, 0), 3)

            # Display the result using Matplotlib
            plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
            st.image(cv2.cvtColor(res, cv2.COLOR_BGR2RGB), use_column_width=True, caption="Image with Extracted License Plate and Text")

# Note: The structure of your application may require adjustments based on your specific requirements.


