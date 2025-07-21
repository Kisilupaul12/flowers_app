import os
import numpy as np
from django.shortcuts import render
from tensorflow.keras.models import load_model
from PIL import Image
from .forms import UploadImageForm

# Load model once at the module level to avoid reloading on every request
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'flower_model.keras')
model = load_model(MODEL_PATH)

# Define the class names in the same order as used during model training
CLASS_NAMES = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

def predict_flower(request):
    prediction = None
    if request.method == 'POST':
        form = UploadImageForm(request.POST, request.FILES)
        if form.is_valid():
            # Load and resize the image to match model's input shape (128x128 in this case)
            img = Image.open(request.FILES['image']).resize((180, 180))

            # Convert image to numpy array and normalize
            img_array = np.array(img) / 255.0

            # Ensure the image has 3 channels (RGB)
            if img_array.shape[-1] != 3:
                img_array = img_array[..., :3]

            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 128, 128, 3)

            # Make prediction
            pred = model.predict(img_array)
            predicted_class = CLASS_NAMES[np.argmax(pred)]

            # Set prediction message
            prediction = f"Predicted: {predicted_class}"
    else:
        form = UploadImageForm()

    return render(request, 'classifier/predict.html', {
        'form': form,
        'prediction': prediction
    })
