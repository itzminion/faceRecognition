from tkinter.tix import Tree
import cv2 as cv 
import os


def makePath(path):
    if not os.path.exists(path):
        os.mkdir(path)
        return 1
    else:
        print("already exists")
        return 0
while True:   
    individual = input("Name of the person : ")
    path = os.getcwd() + "\Resources" + "\{}".format(individual)
    if (not makePath(path)):
        print("Person already exists, or enter full name")
    else:
        break

camera = cv.VideoCapture(0)
face_cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")

count = 0 
while True:
    _, img = camera.read()
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.05,4)
    for (x,y,w,h) in faces:
        cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        file_name = "{}_{}.jpg".format(individual,count)
        count += 1
        cv.imwrite(os.path.join(path,file_name),img[y:y+h,x:x+w])
        cv.imshow("frame cam",img)
    if cv.waitKey(100) & 0xFF == ord('q'):
        break 
    elif count > 100:
        break 
camera.release()
cv.destroyAllWindows()