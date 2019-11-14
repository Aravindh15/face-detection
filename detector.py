# Import OpenCV2 for image processing
import cv2,os



# Import numpy for matrices calculations

import numpy as np
from PIL import Image 



# Create Local Binary Patterns Histograms for face recognization

recognizer = cv2.face.LBPHFaceRecognizer_create()



# Load the trained mode

recognizer.read('trainer/trainer.yml')



# Load prebuilt model for Frontal Face

cascadePath = "haarcascade_frontalface_default.xml"


# Create classifier from prebuilt model

faceCascade = cv2.CascadeClassifier(cascadePath);



# Set the font style

font = cv2.FONT_HERSHEY_SIMPLEX



# Initialize and start the video frame capture

cam = cv2.VideoCapture(0)



while True:

    ret, im =cam.read()

    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

    faces=faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)

    for(x,y,w,h) in faces:

        cv2.rectangle(im,(x-50,y-50),(x+w+50,y+h+50),(225,0,0),2)

        nbr_predicted, conf = recognizer.predict(gray[y:y+h,x:x+w])
        
        if(nbr_predicted==1):

             nbr_predicted='Aravind'
        
        elif(nbr_predicted==5):

             nbr_predicted='Anu'     

        elif(nbr_predicted==4):

             nbr_predicted='gg'
        elif(nbr_predicted==9):
            
             nbr_predicted='unknown' 

        cv2.putText(im,str(nbr_predicted)+"--"+str(conf), (x,y+h),font, 1.1, (0,255,0)) #Draw the text

        cv2.imshow('im',im)

     
    # If 'q' is pressed, close program

    if cv2.waitKey(10) & 0xFF == ord('q'):

        break



# Stop the camera

cam.release()



# Close all windows

cv2.destroyAllWindows()

