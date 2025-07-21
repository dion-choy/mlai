from PIL import Image, ImageOps
import tensorflow as tf

import cv2
import numpy as np
import os
import sys



label = ''

labels=["Broccoli", "Cauliflower","Unknown"]

frame = None

def import_and_predict(image_data, model):
    
        size = (150,150)    
        image = ImageOps.fit(image_data, size, Image.LANCZOS)
        image = image.convert('RGB')
        image = np.asarray(image)
        image = (image.astype(np.float32) / 255.0)

        img_reshape = image[np.newaxis,...]

        prediction = model.predict(img_reshape)
        
        return prediction

model = tf.keras.models.load_model('model.keras')
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# tflite_model = converter.convert()

# # Save the model.
# with open('tflitemodel.tflite', 'wb') as f:
#   f.write(tflite_model)
    
cap = cv2.VideoCapture(0)

if (cap.isOpened()):
    print("Camera OK")
else:
    cap.open()

while (True):
    ret, original = cap.read()

    frame = cv2.resize(original, (224, 224))
    cv2.imwrite(filename='img.jpg', img=original)
    image = Image.open('img.jpg')

    # Display the predictions
    # print("ImageNet ID: {}, Label: {}".format(inID, label))
    prediction = import_and_predict(image, model)
    #print(prediction)

    predict=np.argmax(prediction[0])
    if prediction[0][predict]<0.6:
         predict=2
    
    cv2.putText(original, labels[predict] + str(prediction[0]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imshow("Classification", original)

    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break

cap.release()
frame = None
cv2.destroyAllWindows()
sys.exit()
