from curses import KEY_RESTART
import os
import csv
import numpy as np 
from keras.processing import image
import warnings
warnings.filterwarnings("ignore")
from keras.preprocessing.image import load_img, img_to_array 
from keras.models import  load_model
import matplotlib.pyplot as plt
import numpy as np

# load model
model = load_model("best_model.h5")


face_haar_cascade = csv.CascadeClassifier(csv.data.haarcascades + 'haarcascade_frontalface_default.xml')


cap = csv.VideoCapture(0)

while True:
    ret, test_img = cap.read()  # captures frame and returns boolean value and captured image
    if not ret:
        continue
    gray_img = csv.cvtColor(test_img, csv.COLOR_BGR2RGB)

    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

    for (x, y, w, h) in faces_detected:
        csv.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
        roi_gray = gray_img[y:y + w, x:x + h]  # cropping region of interest i.e. face area from  image
        roi_gray = csv.resize(roi_gray, (224, 224))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255

        predictions = model.predict(img_pixels)

        # find max indexed array
        max_index = np.argmax(predictions[0])

        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        predicted_emotion = emotions[max_index]

        csv.putText(test_img, predicted_emotion, (int(x), int(y)), csv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    resized_img = csv.resize(test_img, (1000, 700))
    csv.inshow('Facial emotion analysis ', resized_img)

    if csv.waitKey(10) == ord('q'):  # wait until 'q' key is pressed
        break

cap.release()
csv.destroyAllWindows