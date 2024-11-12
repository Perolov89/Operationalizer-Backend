from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image, UnidentifiedImageError
import io

model = tf.keras.models.load_model('number_and_symbol_recognizer.keras')

app = Flask(__name__)

def preprocess_image(image_bytes):
    try:
        # Open the image
        img = Image.open(io.BytesIO(image_bytes))
        
        # Convert to grayscale
        if img.mode != 'L':
            img = img.convert('L')
        
        # Resize to 28x28 px
        img = img.resize((28, 28))
        
        # Normalize and reshape the image
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)
        
        return img_array
    except UnidentifiedImageError:
        return None

@app.route('/predict', methods=['POST'])
def predict():
    # Check if the request contains an image file
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    # Get the image file from the request
    file = request.files['file']
    
    # Read the file and preprocess it
    img_array = preprocess_image(file.read())
    
    # Handle the case where the image is invalid
    if img_array is None:
        return jsonify({"error": "Invalid image format"}), 400
    
    # Run the model prediction
    predictions = model.predict(img_array)
    predicted_label = int(np.argmax(predictions, axis=1)[0])  # Get the label as an integer

    # Return the prediction result
    return jsonify({"predicted_label": predicted_label})

if __name__ == '__main__':
    app.run(debug=True)
