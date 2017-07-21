# -*- coding: utf-8 -*-
import cv2, os
import numpy as np

faceCascadePath = 'haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(faceCascadePath)
recognizer = cv2.face.createLBPHFaceRecognizer()


def get_face_detect(imgPath):
    img = cv2.imread(imgPath)
    cropImg = None

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    faces = faceCascade.detectMultiScale(gray)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roiGray = gray[y:y + h, x:x + w]
        cropImg = roiGray
    return cropImg


def get_iamges_and_labels(path):
    imagesPaths = [os.path.join(path, f) for f in os.listdir(path)]
    images = []
    labels = []
    for imagePath in imagesPaths:
        corpedImage = get_face_detect(imagePath)
        if corpedImage is not None:
            images.append(corpedImage)
            label = int(((os.path.splitext(imagePath)))[0].split('_')[1])
            labels.append(label)
            cv2.imshow("Adding faces to traning set...", corpedImage)
            cv2.waitKey(50)
    return images, labels

trainFilePath = '.\Face Recognition using OpenCV\Train'
images, labels = get_iamges_and_labels(trainFilePath)
print("Image and label loaded")

# Perform the training
recognizer.train(images, np.array(labels))
print("Train Done")

testFilePath = ".\Face Recognition using OpenCV\Test\drbost.20_3.jpg"
print(testFilePath)
testImage = get_face_detect(testFilePath)
print(testImage)
cv2.imshow("Display Test Image", testImage)
cv2.waitKey(10000)

#predict an image
nbrPredicted, conf = recognizer.predict(testImage)
nbrActual = int(((os.path.splitext(testFilePath)))[0].split('_')[1])
if nbrActual == nbrPredicted:
    print("{} is Correctly Recognized with confidence {}".format(nbrActual, conf))
else:
    print("{} is Incorrect Recognized as {}".format(nbrActual, nbrPredicted))
cv2.imshow("Recognizing Face", testImage)
cv2.waitKey(10000)
