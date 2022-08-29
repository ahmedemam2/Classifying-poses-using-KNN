import os
from os import walk
import pandas as pd
import cv2
import mediapipe as mp
import sklearn
import dollarpy
from dollarpy import Recognizer, Template, Point

import numpy as np
mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()
ct2 = 0
files = []
landmarksT = []
labelT = []
landmarksTrain = []
labelTrain = []
landmarksnew = []
labelnew = []
list = ['sit','stand', 'test']

def getTrainlandmarks(path,target):
    cap = cv2.VideoCapture(path)
    xl = []
    yl = []
    templ = []
    labellist = []
    while True:
        success, img = cap.read()
        success, frames = cap.read()
        try:
            imgRGB = cv2.cvtColor(frames, cv2.COLOR_BGR2RGB)
        except:
            break
        results = pose.process(imgRGB)
        # print(results.pose_landmarks)
        if results.pose_landmarks:
            mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
            for id, lm in enumerate(results.pose_landmarks.landmark):
                if id == 12:
                    h, w, c = img.shape
                    labellist.append(target)
                    xl.append(lm.x)
                    yl.append(lm.y)
                    templ.append((lm.x, lm.y))
        # cv2.imshow("Image", img)
        # cv2.waitKey(1)
    cap.release()
    cv2.destroyAllWindows()
    return templ,labellist

for i in range(len(list)):
    for arr in os.listdir(list[i]):
        if list[i] == 'test':
            break
        path = list[i] + '/' + arr
        print(path,list[i])
        landmarksT , labelT = getTrainlandmarks(path,list[i])
        landmarksnew = landmarksnew + landmarksT
        labelnew = labelnew + labelT
print(labelnew)
def getTestlandmarks(path):
    cap = cv2.VideoCapture(path)
    xl = []
    yl = []
    templ = []
    labellist = []
    while True:
        success, img = cap.read()
        success, frames = cap.read()
        try:
            imgRGB = cv2.cvtColor(frames, cv2.COLOR_BGR2RGB)
        except:
            break
        results = pose.process(imgRGB)
        # print(results.pose_landmarks)
        if results.pose_landmarks:
            # mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
            for id, lm in enumerate(results.pose_landmarks.landmark):
                if id == 12:
                    h, w, c = img.shape
                    xl.append(lm.x)
                    yl.append(lm.y)
                    templ.append((lm.x, lm.y))
        # cv2.imshow("Image", img)
        # cv2.waitKey(1)
    cap.release()
    cv2.destroyAllWindows()
    return templ
landmarksTest = []
for arr in os.listdir(list[i]):
    path = list[i] + '/' + arr
    landmarksTest  = getTestlandmarks(path)
xl2 = []
yl2 = []
templ2=[]
cap = cv2.VideoCapture(path)
ct = 0
while True:
    success, img = cap.read()
    success, frames = cap.read()
    try:
        imgRGB = cv2.cvtColor(frames, cv2.COLOR_BGR2RGB)
    except:
        break
    results = pose.process(imgRGB)
    # print(results.pose_landmarks)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        # font
        font = cv2.FONT_HERSHEY_SIMPLEX
        # org
        org = (00, 185)

        # fontScale
        fontScale = 1

        # Red color in BGR
        color = (0, 0, 255)

        # Line thickness of 2 px
        thickness = 2

        X_train = landmarksnew
        X_test = landmarksTest
        y_train = labelnew
        from sklearn.neighbors import KNeighborsClassifier

        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X_train, y_train)
        prediction = knn.predict(X_test)
        # print(knn.predict(X_test))
        knn = prediction[ct]
        ct = ct+1
        text = knn
      # Using cv2.putText() method
        img = cv2.putText(img, text, org, font, fontScale,
                            color, thickness, cv2.LINE_AA, False)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()
