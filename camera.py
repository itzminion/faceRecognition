import cv2
import os
import numpy as np
import pickle
import mediapipe as mp
import cvzone

mp_drawing = mp.solutions.drawing_utils
mp_face = mp.solutions.face_detection.FaceDetection(model_selection=1,min_detection_confidence=0.5)
width=640
height=480

CascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(CascadePath)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('bin/trained.yml')
with open('bin/names.data', 'rb') as filehandle:
            people = pickle.load(filehandle)

def obj_data(img):
    image_input = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    results = faceCascade.detectMultiScale(image_input,1.1,5)
    for (x, y , w, h ) in results:
        faces_roi = image_input[y:y+h , x:x+w]
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2) 
        predict_label, confidence = recognizer.predict(image_input[y:y+h, x:x+w])
        name = people[predict_label]
            
        if confidence <= 50:
            print("Recognized {} with {} confidence".format(name,confidence))
            cv2.putText(img, str(name), (x,y), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
        else:
            print("Recognized an {} person with {}".format(name,confidence))
            cv2.putText(img, "Unknown", (x,y), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
   
        

class VideoCamera(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.video = cv2.VideoCapture(0)
        #self.video = cv2.resize(self.video,(840,640))
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        success, image = self.video.read()
#        image=cv2.resize(image,(840,640))
        obj_data(image)
        
       
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()
