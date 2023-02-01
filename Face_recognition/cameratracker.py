'''from base64 import encode
from sqlite3 import Date
from xmlrpc.client import DateTime'''
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import mysql.connector as m

conn=m.connect(user='root',password='Home#6090',host='localhost',database='cmr')
csr=conn.cursor()
csr.execute('use cmr')
csr.execute('drop table attendance')
csr.execute('create table Attendance(Name varchar(25),Time varchar(10))')
nameList=[]


path='Images'
images=[]
classNames=[]
myList=os.listdir(path)
for cls in myList:
    current=cv2.imread(f'{path}/{cls}')
    images.append(current)
    classNames.append(os.path.splitext(cls)[0])
#print(classNames)


def encodings(images):
    encoded=[]
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encoded.append(encode)
    return encoded


def Attendance(name):
    if name not in nameList:
        now=datetime.now()
        dtstring=now.strftime('%H:%M:%S')
        q1='insert into Attendance(Name,Time)values(%s,%s)'
        v1=(name,dtstring)
        csr.execute(q1,v1)
        conn.commit()
        nameList.append(name)



encodedList=encodings(images)
print('Encoding:Done...')

capture=cv2.VideoCapture(0)
while True:
    check,img=capture.read()
    simg=cv2.resize(img,(0,0),None,0.25,0.25)
    simg=cv2.cvtColor(simg,cv2.COLOR_BGR2RGB)
    allfaces=face_recognition.face_locations(simg)
    encodes_current=face_recognition.face_encodings(simg,allfaces)

    for encodeFace,faceLocations in zip(encodes_current,allfaces):
        matches=face_recognition.compare_faces(encodedList,encodeFace)
        FaceDis=face_recognition.face_distance(encodedList,encodeFace)
        
        matchIndex=np.argmin(FaceDis)

        if matches[matchIndex]:
            name=classNames[matchIndex].upper()
            y1,x2,y2,x1=faceLocations
            y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_DUPLEX,1,(255,255,255),2)
            Attendance(name)

    cv2.imshow('Camera',img)
    cv2.waitKey(1)




