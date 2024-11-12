import tensorflow as tf
import numpy as np
from PIL import Image
import sys

model = tf.keras.models.load_model('number_and_symbol_recognizer.keras')



def predict_single_image(image_path):
    # Load image
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize((28, 28))

    # Convert to arrat and normalize
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    # Predict with the model
    prediction = model.predict(img_array)
    # The predict() method outputs a list of probabilities for each class.
    predicted_label = np.argmax(prediction, axis=1)[0]
    # Returns the index of the highest probability

    # Print and return the result
    print(f"Predicted label: {predicted_label}")
    return predicted_label


if __name__ == "__main__":
    # Pass an image path as an argument when running the script
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        predict_single_image(image_path)
    else:
        print("Bad path to image")