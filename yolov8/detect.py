from ultralytics import YOLO
from PIL import Image, ImageOps
import tensorflow as tf

import cv2
import numpy as np
import os
import sys


model = YOLO("./runs/classify/train3/weights/best.pt")

# For a single image

label = ''

frame = None

def import_and_predict(image_data, model):
    
        size = (100,100)    
        image = ImageOps.fit(image_data, size, Image.LANCZOS)
        image = image.convert('RGB')
        image = np.asarray(image)
        image = (image.astype(np.float32) / 255.0)

        img_reshape = image[np.newaxis,...]

        prediction = model.predict(img_reshape)
        
        return prediction
    
cap = cv2.VideoCapture(0)

if (cap.isOpened()):
    print("Camera OK")
else:
    cap.open()

while (True):
    ret, original = cap.read()

    frame = cv2.resize(original, (224, 224))
    cv2.imwrite(filename='img.jpg', img=original)

    prediction = model.predict(source="./img.jpg", save=True)[0].probs
    print(prediction.data)
    print()

    if np.argmax(prediction.data) == 0:
        predict="It is a broccoli!"
    elif np.argmax(prediction.data) == 1:
        predict="It is a cauliflower!"
    else:
        predict="It is a unknown!"
    
    cv2.putText(original, predict, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(original, str(prediction.data), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow("Classification", original)

    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break;

cap.release()
frame = None
cv2.destroyAllWindows()
sys.exit()
