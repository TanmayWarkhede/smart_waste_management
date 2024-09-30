from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

model = tf.keras.models.load_model('model/waste_management_model.h5')

# List of object names corresponding to the class indices
class_labels = [
    "Cardboard", "Glass", "Metal", "Paper", "Plastic", "Trash", 
    "Organic", "E-waste", "Textiles", "Wood", "Food", "Others"
]

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        
        if file:
            # Open the image file
            image = Image.open(io.BytesIO(file.read()))
            # Preprocess the image
            image = image.convert('RGB')  # Ensure the image is in RGB format
            image = image.resize((224, 224))
            image = np.array(image) / 255.0
            image = np.expand_dims(image, axis=0)
            
            # Predict using the model
            predictions = model.predict(image)
            class_index = np.argmax(predictions[0])
            class_label = class_labels[class_index]  # Get the name of the detected object

            return jsonify({'class': class_label})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
