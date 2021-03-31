import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np

import serial
import time

arduino = serial.Serial(port='COM3', baudrate=9600, timeout=.1)


def write_read(x):
    arduino.write(int(x))
    time.sleep(0.05)


model = load_model('masknet.h5')

labels_dict = {1: 'without_mask', 0: 'with_mask'}
color_dict = {1: (0, 0, 255), 0: (0, 255, 0)}

size = 4
webcam = cv2.VideoCapture(0)

classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    (rval, im) = webcam.read()
    im = cv2.flip(im, 1, 1)

    mini = cv2.resize(im, (im.shape[1] // size, im.shape[0] // size))

    faces = classifier.detectMultiScale(mini)

    for f in faces:
        (x, y, w, h) = [v * size for v in f]

        face_img = im[y:y + h, x:x + w]
        resized = cv2.resize(face_img, (128, 128))
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1, 128, 128, 3))
        reshaped = np.vstack([reshaped])
        result = model.predict(reshaped)
        label = np.argmax(result, axis=1)[0]

        write_read(label)

        cv2.rectangle(im, (x, y), (x + w, y + h), color_dict[label], 2)
        cv2.rectangle(im, (x, y - 40), (x + w, y), color_dict[label], -1)
        cv2.putText(im, labels_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow('LIVE', im)
    key = cv2.waitKey(10)

    if key == 27:
        break

webcam.release()

cv2.destroyAllWindows()
