# Testing the model wiht external images that are not from the dataset

import tensorflow as tf
import numpy as np
from PIL import Image, UnidentifiedImageError
import sys
import os

model = tf.keras.models.load_model('number_and_symbol_recognizer.keras')

def predict_single_image(image_path):
    try:
        # Check if file exists
        if not os.path.isfile(image_path):
            print(f"Error: The file '{image_path}' does not exist.")
            return None

        # Attempt to open the image
        img = Image.open(image_path)

        # Convert to grayscale if necessary
        if img.mode != 'L':
            img = img.convert('L')

        # Resize to the required input size
        img = img.resize((28, 28))

        # Convert to numpy array and normalize
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        # Check that the reshaped array has the correct dimensions
        if img_array.shape != (1, 28, 28, 1):
            print("Error: Image shape is incorrect after processing.")
            return None

        # Make a prediction
        prediction = model.predict(img_array)
        predicted_label = np.argmax(prediction, axis=1)[0]

        # Output the result
        print(f"Predicted label: {predicted_label}")
        return predicted_label

    except UnidentifiedImageError:
        print(f"Error: The file '{image_path}' is not a valid image.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


if __name__ == "__main__":
    # Pass an image path as an argument when running the script
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        predict_single_image(image_path)
    else:
        print("Please provide the path to an image.")
