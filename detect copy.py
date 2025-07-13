from PIL import Image, ImageOps
import tensorflow as tf

import cv2
import numpy as np
import os
import sys



label = ''

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

model = tf.keras.models.load_model('./models/gen3/model.keras')



image = Image.open('broccoli.jpg')

# Display the predictions
# print("ImageNet ID: {}, Label: {}".format(inID, label))
prediction = import_and_predict(image, model)
#print(prediction)

if np.argmax(prediction) == 0:
    predict="It is a broccoli!"
elif np.argmax(prediction) == 1:
    predict="It is a cauliflower!"
else:
    predict="It is a unknown!"
print(prediction)




cv2.destroyAllWindows()
sys.exit()
