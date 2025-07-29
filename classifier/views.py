import os
import numpy as np
from django.shortcuts import render
from django.http import JsonResponse
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings
from PIL import Image
from .forms import UploadImageForm
import uuid

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
        
        # Use Django's BASE_DIR to get project root directory
        MODEL_PATH = os.path.join(settings.BASE_DIR, 'classifier', 'flower_model (1).keras')
        
        # Check if model file exists
        if not os.path.exists(MODEL_PATH):
            print(f"Model file not found at: {MODEL_PATH}")
            # Also check alternative locations as fallback
            fallback_locations = [
                os.path.join(settings.BASE_DIR, 'flower_model (1).keras'),  # Project root
                os.path.join(os.path.dirname(os.path.abspath(__file__)), 'flower_model (1).keras'),  # App directory
            ]
            
            for fallback_path in fallback_locations:
                if os.path.exists(fallback_path):
                    MODEL_PATH = fallback_path
                    print(f"Found model at fallback location: {MODEL_PATH}")
                    break
            else:
                print(f"Model not found in any of the expected locations")
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

def save_uploaded_image(uploaded_file):
    """Save uploaded image and return the URL"""
    try:
        # Generate unique filename
        file_extension = uploaded_file.name.split('.')[-1].lower()
        unique_filename = f"{uuid.uuid4()}.{file_extension}"
        
        # Save file
        file_path = default_storage.save(f'uploads/{unique_filename}', ContentFile(uploaded_file.read()))
        
        # Return URL for the saved file
        file_url = default_storage.url(file_path)
        return file_url, file_path
        
    except Exception as e:
        print(f"Error saving image: {e}")
        return None, None

def predict_flower(request):
    prediction = None
    error_message = None
    image_url = None
    confidence_score = None
    
    # Try to load model
    current_model = load_model_safe()
    
    if current_model is None:
        error_message = "Model is not available. Please check if the model file exists and TensorFlow is properly installed."
    
    if request.method == 'POST':
        form = UploadImageForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                uploaded_file = request.FILES['image']
                
                # Validate file type
                allowed_extensions = ['jpg', 'jpeg', 'png', 'bmp', 'gif']
                file_extension = uploaded_file.name.split('.')[-1].lower()
                
                if file_extension not in allowed_extensions:
                    error_message = f"Please upload a valid image file. Allowed formats: {', '.join(allowed_extensions)}"
                else:
                    # Save the uploaded image
                    image_url, file_path = save_uploaded_image(uploaded_file)
                    
                    if image_url is None:
                        error_message = "Failed to save uploaded image."
                    elif current_model is not None:
                        # Reset file pointer for processing
                        uploaded_file.seek(0)
                        
                        # Load and resize the image to match model's input shape (180x180)
                        img = Image.open(uploaded_file).convert('RGB').resize((180, 180))

                        # Convert image to numpy array and normalize
                        img_array = np.array(img) / 255.0

                        # Add batch dimension
                        img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 180, 180, 3)

                        # Make prediction
                        pred = current_model.predict(img_array, verbose=0)
                        predicted_class = CLASS_NAMES[np.argmax(pred)]
                        confidence = float(np.max(pred))
                        confidence_score = confidence

                        # Set prediction message with confidence
                        prediction = f"Predicted: {predicted_class.title()}"
                        
                        # Add confidence level description
                        if confidence > 0.9:
                            confidence_desc = "Very High"
                        elif confidence > 0.7:
                            confidence_desc = "High"
                        elif confidence > 0.5:
                            confidence_desc = "Moderate"
                        else:
                            confidence_desc = "Low"
                            
                    else:
                        error_message = "Model is not available for prediction."
                        
            except Exception as e:
                error_message = f"Error during prediction: {str(e)}"
                print(f"Prediction error: {e}")
    else:
        form = UploadImageForm()

    # Prepare context
    context = {
        'form': form,
        'prediction': prediction,
        'error_message': error_message,
        'image_url': image_url,
        'confidence_score': confidence_score,
        'confidence_percentage': f"{confidence_score:.1%}" if confidence_score else None,
    }

    return render(request, 'classifier/predict.html', context)

# Optional: API endpoint for predictions
def api_predict(request):
    """API endpoint for programmatic access"""
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST method allowed'}, status=405)
    
    current_model = load_model_safe()
    if current_model is None:
        return JsonResponse({'error': 'Model not available'}, status=503)
    
    try:
        form = UploadImageForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = request.FILES['image']
            
            # Process image
            img = Image.open(uploaded_file).convert('RGB').resize((180, 180))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Make prediction
            pred = current_model.predict(img_array, verbose=0)
            predicted_class = CLASS_NAMES[np.argmax(pred)]
            confidence = float(np.max(pred))
            
            return JsonResponse({
                'prediction': predicted_class,
                'confidence': confidence,
                'all_predictions': {
                    CLASS_NAMES[i]: float(pred[0][i]) 
                    for i in range(len(CLASS_NAMES))
                }
            })
        else:
            return JsonResponse({'error': 'Invalid form data'}, status=400)
            
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
