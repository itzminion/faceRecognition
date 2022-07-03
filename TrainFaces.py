from asyncio.sslproto import _create_transport_context
from cProfile import label
from csv import Dialect
import os 
import cv2 as cv 
import numpy as np 

people = []
for i in os.listdir(r'C:\Users\sreelal\Documents\Project\openCV-py\Resources'):
    people.append(i)

DIR = r'C:\Users\sreelal\Documents\Project\openCV-py\Resources'

haar_cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")

features = []
labels = [] 

def create_train():
    for person in people:
        path = os.path.join(DIR, person )
        label = people .index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path,img)

            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            facesRect = haar_cascade.detectMultiScale(gray,1.1,4)

            for ( x, y , w, h) in facesRect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)  

create_train()

print('Training done ---------------------')

features = np.array(features, dtype='object')
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()

face_recognizer.train(features,labels)


face_recognizer.save(r'bin/face_trained.yml')

np.save(r'bin/features.npy', features)
np.save(r'bin/labels.npy', labels)
