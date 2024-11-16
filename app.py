from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image, UnidentifiedImageError, ImageOps
import io
import base64

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
            (img_array[0, :, :, 0] * 255).astype(np.uint8)).save("processed_image.png")

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


@app.route('/predict', methods=['POST'])
def predict():

    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({"error": "No image data provided"}), 400

    # Extract base64-encoded string from JSON and remove "data:image/png;base64,"
    base64_data = data['image'].split(
        ',')[1] if ',' in data['image'] else data['image']

    try:
        image_bytes = base64.b64decode(base64_data)

        # Preprocess the image
        img_array = preprocess_image(image_bytes)

        if img_array is None:
            return jsonify({"error": "Invalid image format"}), 400

        # Run model
        predictions = model.predict(img_array)
        predicted_label = np.argmax(predictions, axis=1)[0]

        # Convert to a symbol or number
        predicted_symbol = convert_to_symbol(predicted_label)

        # Return both the numerical label and the symbol
        return jsonify({
            "predicted_label": int(predicted_label),
            "predicted_symbol": predicted_symbol,
            "confidence": float(predictions[0][predicted_label])
        })

    except (ValueError, base64.binascii.Error):
        return jsonify({"error": "Invalid base64 data"}), 400


if __name__ == '__main__':
    app.run(debug=True)
