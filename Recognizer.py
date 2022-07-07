import cv2 as cv
import os
import numpy as np 
import pickle

CascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv.CascadeClassifier(CascadePath)
recognizer = cv.face.LBPHFaceRecognizer_create()

camera = cv.VideoCapture(0)

recognizer.read('bin/trained.yml')

with open('bin/names.data', 'rb') as filehandle:
    people = pickle.load(filehandle)

print(people)
while True:
    _, img = camera.read()
    imgGray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imgGray,1.1,5)
    for (x, y , w, h ) in faces:
        faces_roi = imgGray[y:y+h , x:x+w]
        cv.rectangle(img, (x,y), (x+w, y+h), (0,0,255), thickness=2)
        predict_label, confidence = recognizer.predict(imgGray[y:y+h, x:x+w])
        name = people[predict_label]
            
        if confidence <= 50:
            print("Recognized {} with {} confidence".format(name,confidence))
            cv.putText(img, str(name), (x,y), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
        else:
            print("Recognized an Unknown person")
            cv.putText(img, "Unknown", (x,y), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
    cv.imshow("Stream",img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    