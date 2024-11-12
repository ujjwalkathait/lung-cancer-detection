import os
import json
from PIL import Image

import numpy as np
import tensorflow as tf

working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/models/model_lung_lowData.keras"

model = tf.keras.models.load_model(model_path)

# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    # Load the image
    img = Image.open(image_path)
    # Resize the image
    img = img.resize(target_size)
    # Convert the image to a numpy array
    img_array = np.array(img)
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    # Scale the image values to [0, 1]
    img_array = img_array.astype('float32') / 255.
    return img_array


# Function to Predict the Class of an Image
def predict_image_class(model, image_path):
    # preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(image_path)
    # predicted_class_index = np.argmax(predictions, axis=1)[0]
    # predicted_class_name = class_indices[str(predicted_class_index)]
    # return predicted_class_name

    return predictions

image_path = 'Test cases/000001_03_01_088.png'
predicted_class_name = predict_image_class(model, image_path)

# Output the result
print("Predicted Class Name:", predicted_class_name)