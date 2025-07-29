import os
import numpy as np
from django.shortcuts import render
from django.http import JsonResponse
from PIL import Image
from .forms import UploadImageForm

# Define the class names in the same order as used during model training
CLASS_NAMES = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

# Global variable for model
model = None

def load_model_safe():
    """Safely load the model with error handling"""
    global model
    if model is not None:
        return model
    
    try:
        from tensorflow.keras.models import load_model
        
        BASE_DIR = os.path.dirname(os.path.abspath(_file_))
        MODEL_PATH = os.path.join(BASE_DIR, 'flower_model.keras')
        
        # Check if model file exists
        if not os.path.exists(MODEL_PATH):
            print(f"Model file not found at: {MODEL_PATH}")
            return None
        
        print(f"Loading model from: {MODEL_PATH}")
        
        # Try loading with different approaches
        try:
            model = load_model(MODEL_PATH)
            print("Model loaded successfully with default settings")
        except Exception as e:
            print(f"Failed to load with default settings: {e}")
            # Try loading without compilation for compatibility
            model = load_model(MODEL_PATH, compile=False)
            print("Model loaded successfully without compilation")
        
        return model
        
    except ImportError:
        print("TensorFlow not available")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def predict_flower(request):
    prediction = None
    error_message = None
    
    # Try to load model
    current_model = load_model_safe()
    
    if current_model is None:
        error_message = "Model is not available. Please check if the model file exists and TensorFlow is properly installed."
    
    if request.method == 'POST':
        form = UploadImageForm(request.POST, request.FILES)
        if form.is_valid() and current_model is not None:
            try:
                # Load and resize the image to match model's input shape (180x180)
                img = Image.open(request.FILES['image']).convert('RGB').resize((180, 180))

                # Convert image to numpy array and normalize
                img_array = np.array(img) / 255.0

                # Add batch dimension
                img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 180, 180, 3)

                # Make prediction
                pred = current_model.predict(img_array, verbose=0)
                predicted_class = CLASS_NAMES[np.argmax(pred)]
                confidence = float(np.max(pred))

                # Set prediction message with confidence
                prediction = f"Predicted: {predicted_class} (Confidence: {confidence:.2%})"
                
            except Exception as e:
                error_message = f"Error during prediction: {str(e)}"
                print(f"Prediction error: {e}")
        
        elif form.is_valid() and current_model is None:
            error_message = "Model is not available for prediction."
    else:
        form = UploadImageForm()

    return render(request, 'classifier/predict.html', {
        'form': form,
        'prediction': prediction,
        'error_message': error_message
    })
