import cv2
import streamlit as st
import numpy as np
import imutils
import easyocr
import matplotlib.pyplot as plt
st.title("License Plate Detection App")

# Upload an image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
font = cv2.FONT_HERSHEY_SIMPLEX

if uploaded_image is not None:
    # Load the image
    img = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)
    conv_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image(conv_img, use_column_width=True, caption="Original Image")

    # Convert the image to grayscale
    img_gray = cv2.cvtColor(conv_img, cv2.COLOR_BGR2GRAY)

    # Load the helmet cascade classifier
    plate_cascade = cv2.CascadeClassifier(r'D:\DL_Projects\helmet_detection\haarcascade_helmet.xml')

    # Detect plates
    for i in range(1, 100):
        plates = plate_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=i, minSize=(60, 60))
        if len(plates) == 1:
            (x, y, w, h) = plates[0]
            cv2.rectangle(img, (x, y), (x + 3 * w, y + 3 * h), color=(0, 255,0), thickness=2)
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_column_width=True, caption="Image with Detected Helmet")
            break

    if len(plates) == 0:
        # Your second code section (executed only if no plates are detected)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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
            new_image = cv2.bitwise_and(img, img, mask=mask)

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
            res = cv2.rectangle(img, tuple(location[0][0]),tuple(location[2][0]), (0, 255,0), 3)
        # Display the result using Matplotlib
            plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
            st.image(cv2.cvtColor(res, cv2.COLOR_BGR2RGB), use_column_width=True,
                 caption="Image with Extracted License Plate and Text")
plt.show()
