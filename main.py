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
def getpath():
    files = []
    for arr in os.listdir('VIDEOS'):
        files.append(arr)
    return files

files = getpath()
cap = cv2.VideoCapture("VIDEOS/sit/s1.mp4")
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
                labellist.append("sit")
                xl.append(lm.x)
                yl.append(lm.y)
                templ.append((lm.x,lm.y))
    # cv2.imshow("Image", img)
    # cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()
cap = cv2.VideoCapture("VIDEOS/sit/sit_1.mp4")
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
            h, w, c = img.shape
            if id == 12:
                labellist.append("sit")
                xl.append(lm.x)
                yl.append(lm.y)
                templ.append((lm.x,lm.y))

    # cv2.imshow("Image", img)
    # cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()

cap = cv2.VideoCapture("VIDEOS/stand/1.mp4")
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
            h, w, c = img.shape
            if id == 12:
                labellist.append("stand")
                xl.append(lm.x)
                yl.append(lm.y)
                templ.append((lm.x,lm.y))
    # cv2.imshow("Image", img)
    # cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()

cap = cv2.VideoCapture("VIDEOS/stand/2.mp4")
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
            h, w, c = img.shape
            if id == 12:
                labellist.append('stand')
                xl.append(lm.x)
                yl.append(lm.y)
                templ.append((lm.x, lm.y))

    # cv2.imshow("Image", img)
    # cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()
xl2 = []
yl2 = []
templ2=[]
cap = cv2.VideoCapture("VIDEOS/test.mp4")
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
            h, w, c = img.shape
            if id == 12:
                xl2.append(lm.x)
                yl2.append(lm.y)
                templ2.append((lm.x, lm.y))

    # cv2.imshow("Image", img)
    # cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()

unknownlabel = []
cap = cv2.VideoCapture("VIDEOS/test.mp4")
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
        text = 'hi'
        # org
        org = (00, 185)

        # fontScale
        fontScale = 1

        # Red color in BGR
        color = (0, 0, 255)

        # Line thickness of 2 px
        thickness = 2

        X_train = templ
        X_test = templ2
        y_train = labellist
        from sklearn.neighbors import KNeighborsClassifier

        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X_train, y_train)
        preiction = knn.predict(X_test)
        # print(knn.predict(X_test))
        knn = preiction[ct]
        ct = ct+1
        text = knn
      # Using cv2.putText() method
        img = cv2.putText(img, text, org, font, fontScale,
                            color, thickness, cv2.LINE_AA, False)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()

# from sklearn.model_selection import train_test_split
# X = templ
# y = labellist
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
# X_train = templ
# X_test = templ2
# y_train = labellist
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler
#
# from sklearn.linear_model import LogisticRegression, RidgeClassifier
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# pipelines = {
#     'lr':make_pipeline(StandardScaler(), LogisticRegression()),
#     'rc':make_pipeline(StandardScaler(), RidgeClassifier()),
#     'rf':make_pipeline(StandardScaler(), RandomForestClassifier()),
#     'gb':make_pipeline(StandardScaler(), GradientBoostingClassifier()),
# }
# fit_models = {}
# for algo, pipeline in pipelines.items():
#     model = pipeline.fit(X_train, y_train)
#     fit_models[algo] = model
#
# from sklearn.neighbors import KNeighborsClassifier
# knn = KNeighborsClassifier(n_neighbors=3)
# knn.fit(X_train, y_train)
# preiction = knn.predict(X_test)
# print(knn.predict(X_test))
# print(knn.score(X_test,y))
# Exporting dataframe to excel
# df = pd.DataFrame({
#                    'label' : preiction
#                    })
# writer = pd.ExcelWriter('output.xlsx')
# df.to_excel(writer)
#
# # save the excel
# writer.save()
# print("DataFrame is exported successfully to 'converted-to-excel.xlsx' Excel File.")
