from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image, UnidentifiedImageError, ImageOps
import io
import base64
from dotenv import load_dotenv
import os

load_dotenv()

serving_endpoint = os.getenv("ENDPOINT")

model = tf.keras.models.load_model('number_and_symbol_recognizer.keras')

class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
               'add', 'divide', 'multiply', 'subtract']

app = Flask(__name__)
CORS(app)


def preprocess_image(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes))

        if img.mode != 'L':
            img = img.convert('L')

        img = img.resize((28, 28))

        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        # Check result
        Image.fromarray(
            (img_array[0, :, :, 0] * 255).astype(np.uint8)).save('processed_image.png')

        return img_array
    except UnidentifiedImageError as e:
        print(f"Error processing image: {e}")
        return None


def convert_to_symbol(predicted_label):
    symbol_map = {
        'add': '+',
        'divide': '/',
        'multiply': '*',
        'subtract': '-'
    }

    # Get the class name for the predicted label
    predicted_class = class_names[predicted_label]

    # Convert to symbol if it's an operator, otherwise return the number
    return symbol_map.get(predicted_class, predicted_class)


@app.route(serving_endpoint, methods=['POST'])
def predict():
    # print("Received request")  # debug print

    data = request.get_json()

    # print("Received data:", data) # debug print
    if not data or 'images' not in data:
        return jsonify({'error': 'No image data provided'}), 400

    required_keys = ['firstNumber', 'operator', 'secondNumber']
    if not all(key in data['images'] for key in required_keys):
        return jsonify({'error': 'Missing required keys in images'}), 400

    try:
        predictions = {}
        # print("Images field content:", data['images']) # debug print
        for id, image_data in data['images'].items():
            # Extract base64-encoded string from JSON and remove "data:image/png;base64,"
            base64_data = image_data.split(
                ',')[1] if ',' in image_data else image_data

            image_bytes = base64.b64decode(base64_data)

            # Preprocess the image
            img_array = preprocess_image(image_bytes)
            # print(img_array) # debug print
            if img_array is None:
                return jsonify({'error': f'Invalid image format for {id}'}), 400

            # Run model
            prediction = model.predict(img_array)
            predicted_label = np.argmax(prediction, axis=1)[0]

            # Convert to a symbol or number
            predicted_symbol = convert_to_symbol(predicted_label)

        # Return both the numerical label and the symbol
            predictions[id] = {
                'predicted_label': int(predicted_label),
                'predicted_symbol': predicted_symbol,
                'accuracy': float(prediction[0][predicted_label])
            }
        return jsonify(predictions)

    except (ValueError, base64.binascii.Error) as e:
        return jsonify({'error': f'There was an error processing one or more images: {e}'}), 400


if __name__ == '__main__':
    app.run(debug=True)
