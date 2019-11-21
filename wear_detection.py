# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from gpiozero import LEDBoard
from gpiozero.tools import random_values
from imutils.video import VideoStream
from threading import Thread
import numpy as np
import imutils
import time
import cv2
import os

MODEL_PATH = "final_model.model"

print("[INFO] loading model...")
model = load_model(MODEL_PATH)

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

while True:
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	image = cv2.resize(frame, (28, 28))
	image = image.astype("float") / 255.0
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)

	(shirt, t-shirt) = model.predict(image)[0]
	if t-shirt > shirt:
	    label = "t-shirt"
	    proba = t-shirt
	else:
            label = "shirt"
            proba = shirt

        label = "{}: {:.2f}%".format(label, proba * 100)
        frame = cv2.putText(frame, label, (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
 
        if key == ord("q"):
            break

print("[INFO] cleaning up...")
cv2.destroyAllWindows()
vs.stop()
