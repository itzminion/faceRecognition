import cv2 as cv 
import os
import pickle
from cv2 import imread 
import numpy as np 

people = []
DIR = os.getcwd() + r"\Resources"
for i in os.listdir(DIR):
    if os.path.isdir(os.path.join(DIR,i)):
      people.append(i)



cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv.CascadeClassifier(cascadePath)
recognizer = cv.face.LBPHFaceRecognizer_create()


def get_images_and_labels():
    images = []
    labels = []
    names = []
    for person in people:
        path = os.path.join(DIR, person)
        name = person
        label = people.index(person)
        image_paths = [os.path.join(path, f) for f in os.listdir(path)]
        names.append(name)
        for image_path in image_paths:
            img = cv.imread(image_path)
            print(image_path)
            imgGray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(imgGray)
            for ( x, y, w , h) in faces:
                images.append(imgGray[y:y+h, x:x+w])
                labels.append(label)
            
        print("Training {} completed.... ".format(person))
    return images,labels,names
    
images, labels, names = get_images_and_labels()
cv.destroyAllWindows()

recognizer.train(images,np.array(labels))
recognizer.save(r"bin/trained.yml")

with open('bin/names.data', 'wb') as filehandle:
    pickle.dump(names, filehandle)

print("\nTraing Succesfully Completed....")