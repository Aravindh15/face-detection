import requests
import cv2
import numpy as np
from PIL import Image
import os
from gpiozero import LED
from time import sleep

led=LED(17)
led1=LED(18)

def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)
# Create Local Binary Patterns Histograms for face recognization
recognizer = cv2.face.LBPHFaceRecognizer_create()

assure_path_exists("trainer/")

# Load the trained mode
recognizer.read('trainer/trainer.yml')

# Load prebuilt model for Frontal Face
cascadePath = "haarcascade_frontalface_default.xml"

# Create classifier from prebuilt model
faceCascade = cv2.CascadeClassifier(cascadePath);

# Set the font style
font = cv2.FONT_HERSHEY_SIMPLEX


url = "http://192.168.43.1:8080/shot.jpg"

while True:

    img_resp = requests.get(url)
    img_arr = np.array(bytearray(img_resp.content),dtype=np.uint8)
    im = cv2.imdecode(img_arr, -1)
    
    # Convert the captured frame into grayscale
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

    # Get all face from the video frame
    faces=faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)


    # For each face in faces
    for(x,y,w,h) in faces:

        cv2.rectangle(im,(x-50,y-50),(x+w+50,y+h+50),(225,0,0),2)

        nbr_predicted, conf = recognizer.predict(gray[y:y+h,x:x+w])
        
        if(nbr_predicted==1):

             nbr_predicted='Aravind'
             led1.off()
             led.on()
             
        elif(nbr_predicted==2):

             nbr_predicted='Latha'
             led1.off()
             led.on()


        else:
            
             nbr_predicted='unknown'
             led.off()
             led1.on()

        cv2.putText(im,str(nbr_predicted)+"--"+str(conf), (x,y+h),font, 1.1, (0,255,0)) #Draw the text

        

    cv2.imshow("AndroidCam", im)

    # If 'q' is pressed, close program
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Stop the camera
cam.release()


# Close all windows
cv2.destroyAllWindows()


   
