
import tensorflow as tf
from PIL import Image,ImageOps
import numpy as np



model = tf.keras.models.load_model('./models/gen7/model.hdf5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open('tflitemodel.tflite', 'wb') as f:
  f.write(tflite_model)



# test
import numpy as np
import tensorflow as tf

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="/home/me/AndroidStudioProjects/tensorflowdemo/app/src/main/ml/tflitemodel.tflite")

interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
img_path = "sampleImg.jpg"
image=Image.open(img_path)
image = ImageOps.fit(image, (100, 100), method=Image.BILINEAR, centering=(0.5, 0.5))
image = np.asarray(image).astype(np.float32) / 255.0
image = np.expand_dims(image, axis=0)




interpreter.set_tensor(input_details[0]['index'], image)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)