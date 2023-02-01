from base64 import encode
import cv2
import numpy as np
import face_recognition



imgElon=face_recognition.load_image_file('Images/ElonMusk.jpg')
#conversion of images to rgb format(library uses only RGB):
imgElon=cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)
imgElonR=face_recognition.load_image_file('Images/SundarPichai.jpg')
imgElonR=cv2.cvtColor(imgElonR,cv2.COLOR_BGR2RGB)

#Maps coordinates of corners of faces using deep learning:
faceLocation=face_recognition.face_locations(imgElon)[0]
#Encoding face:
encodeElon=face_recognition.face_encodings(imgElon)[0]
#Draws barrier over face coordinates
cv2.rectangle(imgElon,(faceLocation[3],faceLocation[0],faceLocation[1],faceLocation[2]),(255,0,255),2)

#Maps coordinates of corners of faces using deep learning:
faceLocationR=face_recognition.face_locations(imgElonR)[0]
#Encoding face:
encodeElonR=face_recognition.face_encodings(imgElonR)[0]
#Draws barrier over face coordinates
cv2.rectangle(imgElonR,(faceLocationR[3],faceLocationR[0],faceLocationR[1],faceLocationR[2]),(255,0,255),2)


#Checks for similaraties between faces:
results=face_recognition.compare_faces([encodeElon],encodeElonR)
FaceDistance=face_recognition.face_distance([encodeElon],encodeElonR)
print(results,FaceDistance)
cv2.putText(imgElonR,f'{results} {round(FaceDistance[0],2)}',(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)



cv2.imshow('Elon Musk',imgElon)
cv2.imshow('Elon Musk Reference',imgElonR)
cv2.waitKey(0)

