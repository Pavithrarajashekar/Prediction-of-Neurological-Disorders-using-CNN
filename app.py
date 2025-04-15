import os
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import GoogLeNet_Weights
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Device configuration
device = torch.device("cpu")

# Path to the trained model
model_path = "C:/Users/PAVITHRA R/Downloads/googlenet_model_run_5.pth"
# Define the model and adjust for the number of classes
num_classes = 5  # Replace with the number of classes in your dataset
model = models.googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# Load the saved model weights
checkpoint = torch.load(model_path, map_location=device)

# Extract state_dict and load into the model
if "state_dict" in checkpoint:
    model.load_state_dict(checkpoint["state_dict"])
else:
    model.load_state_dict(checkpoint)

# Ensure the model is in evaluation mode
model.eval()

# Extract class names if available
class_names = checkpoint.get("class_names", ["Alzheimers", "Glioma", "Meningioma", "Normal", "Parkinson"])  # Replace default with your labels

# Define image preprocessing transformations
transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Route for the main page
@app.route('/')
def index():
    return render_template('index.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected image'}), 400

    try:
        # Load and preprocess the image
        image = Image.open(file).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)

        # Perform prediction
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            class_id = predicted.item()
            class_name = class_names[class_id]

        return jsonify({'class_id': class_id, 'class_name': class_name})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
