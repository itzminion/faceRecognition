from tkinter.tix import Tree
import cv2 as cv 
import os 

def makePath(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

camera = cv.VideoCapture(1)
face_cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")

face_id = 1 
count = 0 
makePath("dataset/")
while True:
    _, img = camera.read()
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.05,4)
    for (x,y,w,h) in faces:
        cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        count += 1
        cv.imwrite("dataset/User_" + str(face_id) + "_" + str(count) + ".jpg",img[y:y+h,x:x+w])
        cv.imshow("frame cam",img)
    if cv.waitKey(100) & 0xFF == ord('q'):
        break 
    elif count > 100:
        break 
camera.release()
cv.destroyAllWindows()