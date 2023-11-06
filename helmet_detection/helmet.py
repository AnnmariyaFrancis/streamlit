
import cv2
import matplotlib.pyplot as plt
plate_cascade = cv2.CascadeClassifier(r'D:\DL_Projects\helmet_detection\haarcascade_helmet.xml')
img= cv2.imread(r'D:\DL_Projects\helmet_detection\New folder (2)\without_helmet\IMG_5903.jpg')
conv_img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img_gray= cv2.cvtColor(conv_img, cv2.COLOR_BGR2GRAY)y)
for i in range(1,100):
    plate= plate_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=i,minSize=(60, 60))
    if len(plate)==1:
      (x,y,w,h)=plate[0]
      cv2.rectangle(conv_img,(x, y), (x +4*w, y +4*h),color=(0, 0, 255),thickness=2)
      plt.imshow(conv_img)
      break
    else:
      continue