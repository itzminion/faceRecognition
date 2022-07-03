import cv2

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
camera = cv2.VideoCapture(0)
while True:
    success, img = camera.read()
    faces = faceCascade.detectMultiScale(img, 1.05, 3)
    for (x, y, w, h) in faces:
        #cv2.circle(img, (x+w//2, y+h//2), (w//2),(255,255,0),2)
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
    cv2.imshow("output", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


