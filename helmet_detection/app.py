import cv2
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

plate_cascade = cv2.CascadeClassifier('D:\DL_Projects\helmet_detection\haarcascade_helmet.xml')

st.title("Helmet Detection App")


uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    img = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)
    conv_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


    st.image(conv_img, use_column_width=True, caption="Original Image")


    img_gray = cv2.cvtColor(conv_img, cv2.COLOR_BGR2GRAY)


    if st.button("Detect Helmet"):
        for i in range(1, 100):
            plate = plate_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=i, minSize=(60, 60))
            if len(plate) == 1:
                (x, y, w, h) = plate[0]
                cv2.rectangle(conv_img, (x, y), (x + 4 * w, y + 4 * h), color=(0, 0, 255), thickness=2)
                break


        st.image(conv_img, use_column_width=True, caption="Image with Detected Helmet")


