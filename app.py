from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image, UnidentifiedImageError, ImageOps
import io
import base64

model = tf.keras.models.load_model('number_and_symbol_recognizer.keras')

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
        Image.fromarray((img_array[0, :, :, 0] * 255).astype(np.uint8)).save("processed_image.png")
        
        return img_array
    except UnidentifiedImageError as e:
        print(f"Error processing image: {e}")
        return None


@app.route('/predict', methods=['POST'])
def predict():
    # Ensure JSON payload and "image" key are present
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({"error": "No image data provided"}), 400
    
    # Extract base64-encoded string from JSON and remove "data:image/png;base64,"
    base64_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
    
    try:
        # Decode base64 data
        image_bytes = base64.b64decode(base64_data)
        
        # Preprocess the image
        img_array = preprocess_image(image_bytes)
        
        if img_array is None:
            return jsonify({"error": "Invalid image format"}), 400
        
        # Run model prediction
        predictions = model.predict(img_array)
        predicted_label = int(np.argmax(predictions, axis=1)[0])  # Get the label as an integer
        
        # Return prediction
        return jsonify({"predicted_label": predicted_label})
    
    except (ValueError, base64.binascii.Error):
        return jsonify({"error": "Invalid base64 data"}), 400
    
    

if __name__ == '__main__':
    app.run(debug=True)
