# 1 load xml 2 load jpg 3 haar gray 4 detect 5 draw
import cv2
import numpy as np

# load xml 1 file name
face_xml = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_xml = cv2.CascadeClassifier("haarcascade_eye.xml")
# load jpg
img = cv2.imread('face.jpg')
cv2.imshow('src', img)
# haar gray
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# detect faces 1 data 2 scale 3 5
faces = face_xml.detectMultiScale(gray, 1.3, 5)
print("face=", len(faces))
# draw
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    roi_face = gray[y:y + h, x:x + w]
    roi_color = img[y:y + h, x:x + w]
    # 1 gray
    eyes = eye_xml.detectMultiScale(roi_face)
    print('eye=', len(eyes))
    for (e_x, e_y, e_w, e_h) in eyes:
        cv2.rectangle(roi_color, (e_x, e_y), (e_x + e_w, e_y + e_h), (0, 255, 0), 2)
cv2.imshow('dst', img)
cv2.waitKey(0)
