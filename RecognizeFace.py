from tkinter.tix import Tree
import numpy as np
import os 
import cv2 as cv 

haar_cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")

people = []
for i in os.listdir(r'C:\Users\sreelal\Documents\Project\openCV-py\Resources'):
    people.append(i)

# features = np.load('features.npy')
# labels = np.load('labels.npy') 

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

camera = cv.VideoCapture(0)
while True:
    _, img = camera.read()
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    cv.imshow('Person',gray)

    faces_rect = haar_cascade.detectMultiScale(gray,1.1,4)
    for (x,y,w,h) in faces_rect:
        faces_roi = gray[y:y+h,x:x+w]

        label,confidence = face_recognizer.predict(faces_roi)
        print(f'Label = {label} with a confidence of {confidence}')

        cv.putText(img, str(people[label]), (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
        cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)
    cv.imshow("Detected Face",img)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break 