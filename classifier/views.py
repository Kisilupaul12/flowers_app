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
import threading
import time

# Define the class names in the same order as used during model training
CLASS_NAMES = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

# Global variables for model loading
model = None
model_loading = False
model_error = None

def load_model_in_background():
    """Load model in background thread"""
    global model, model_loading, model_error
    
    if model is not None or model_loading:
        return
    
    model_loading = True
    model_error = None
    
    try:
        print("Starting background model loading...")
        from tensorflow.keras.models import load_model
        
        # Find model path
        MODEL_PATH = os.path.join(settings.BASE_DIR, 'classifier', 'flower_model (1).keras')
        
        if not os.path.exists(MODEL_PATH):
            # Try alternative locations
            fallback_locations = [
                os.path.join(settings.BASE_DIR, 'flower_model (1).keras'),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), 'flower_model (1).keras'),
            ]
            
            for fallback_path in fallback_locations:
                if os.path.exists(fallback_path):
                    MODEL_PATH = fallback_path
                    break
            else:
                raise FileNotFoundError("Model file not found in any expected location")
        
        print(f"Loading model from: {MODEL_PATH}")
        
        # Try multiple loading approaches for compatibility
        loading_methods = [
            # Method 1: Standard loading
            lambda: load_model(MODEL_PATH),
            # Method 2: Without compilation
            lambda: load_model(MODEL_PATH, compile=False),
            # Method 3: With custom objects (for compatibility)
            lambda: load_model(MODEL_PATH, compile=False, safe_mode=False),
        ]
        
        for i, method in enumerate(loading_methods, 1):
            try:
                print(f"Trying loading method {i}...")
                model = method()
                print(f"Model loaded successfully with method {i}!")
                
                # Print model input shape for debugging
                print(f"Model expects input shape: {model.input_shape}")
                
                # If loaded without compilation, compile it
                if not hasattr(model, 'compiled_loss') or model.compiled_loss is None:
                    try:
                        model.compile(
                            optimizer='adam',
                            loss='sparse_categorical_crossentropy',
                            metrics=['accuracy']
                        )
                        print("Model compiled successfully")
                    except Exception as compile_error:
                        print(f"Warning: Could not compile model: {compile_error}")
                        # Model can still work for predictions without compilation
                
                break  # Successfully loaded
                
            except Exception as method_error:
                print(f"Method {i} failed: {method_error}")
                if i == len(loading_methods):
                    # All methods failed
                    raise method_error
                continue
        
    except Exception as e:
        error_msg = str(e)
        print(f"Error loading model: {error_msg}")
        
        # Try to provide more helpful error message
        if "Sequential" in error_msg:
            model_error = "Model format compatibility issue. Try re-saving your model with current TensorFlow version."
        elif "compile" in error_msg.lower():
            model_error = "Model compilation issue. The model file may be corrupted or incompatible."
        elif "not found" in error_msg.lower():
            model_error = "Model file not found. Please check if the file exists and is accessible."
        else:
            model_error = f"Model loading failed: {error_msg}"
            
        print(f"Final error: {model_error}")
    finally:
        model_loading = False

def preprocess_image(uploaded_file):
    """
    Preprocess image to match model's expected input shape: (None, 180, 180, 3)
    """
    try:
        # Open image and convert to RGB
        img = Image.open(uploaded_file).convert('RGB')
        
        # Resize to exactly what the model expects: 180x180
        img = img.resize((180, 180))
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Normalize pixel values to [0, 1] (assuming model was trained this way)
        img_array = img_array.astype(np.float32) / 255.0
        
        # Add batch dimension: (180, 180, 3) -> (1, 180, 180, 3)
        img_array = np.expand_dims(img_array, axis=0)
        
        print(f"Preprocessed image shape: {img_array.shape}")
        return img_array
        
    except Exception as e:
        print(f"Error preprocessing image: {e}")
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
    global model, model_loading, model_error
    
    prediction = None
    error_message = None
    image_url = None
    confidence_score = None
    is_loading = False
    
    # Start loading model in background if not already loaded/loading
    if model is None and not model_loading and model_error is None:
        thread = threading.Thread(target=load_model_in_background)
        thread.daemon = True
        thread.start()
    
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
                    else:
                        # Check model status
                        if model_loading:
                            error_message = "Model is still loading. Please wait a moment and try again."
                            is_loading = True
                        elif model_error:
                            error_message = f"Model loading failed: {model_error}"
                        elif model is None:
                            error_message = "Model is not available. Please refresh the page and try again."
                        else:
                            try:
                                # Reset file pointer for processing
                                uploaded_file.seek(0)
                                
                                # Preprocess the image correctly
                                img_array = preprocess_image(uploaded_file)
                                
                                if img_array is None:
                                    error_message = "Failed to preprocess the image."
                                else:
                                    # Make prediction
                                    pred = model.predict(img_array, verbose=0)
                                    predicted_class = CLASS_NAMES[np.argmax(pred)]
                                    confidence = float(np.max(pred))
                                    confidence_score = confidence

                                    # Set prediction message
                                    prediction = f"Predicted: {predicted_class.title()}"
                                    
                                    print(f"Prediction successful: {predicted_class} ({confidence:.2%})")
                                
                            except Exception as e:
                                error_message = f"Error during prediction: {str(e)}"
                                print(f"Prediction error: {e}")
                        
            except Exception as e:
                error_message = f"Error processing request: {str(e)}"
                print(f"Request processing error: {e}")
    else:
        form = UploadImageForm()
    
    # Check if model is loading for display
    if model_loading:
        is_loading = True

    # Prepare context
    context = {
        'form': form,
        'prediction': prediction,
        'error_message': error_message,
        'image_url': image_url,
        'confidence_score': confidence_score,
        'confidence_percentage': f"{confidence_score:.1%}" if confidence_score else None,
        'is_loading': is_loading,
        'model_ready': model is not None,
    }

    return render(request, 'classifier/predict.html', context)

def model_status(request):
    """AJAX endpoint to check model loading status"""
    global model, model_loading, model_error
    
    status = {
        'loaded': model is not None,
        'loading': model_loading,
        'error': model_error,
    }
    
    return JsonResponse(status)

# Optional: API endpoint for predictions
def api_predict(request):
    """API endpoint for programmatic access"""
    global model, model_loading, model_error
    
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST method allowed'}, status=405)
    
    if model_loading:
        return JsonResponse({'error': 'Model is still loading, please try again'}, status=503)
    
    if model_error:
        return JsonResponse({'error': f'Model loading failed: {model_error}'}, status=503)
    
    if model is None:
        return JsonResponse({'error': 'Model not available'}, status=503)
    
    try:
        form = UploadImageForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = request.FILES['image']
            
            # Preprocess image correctly
            img_array = preprocess_image(uploaded_file)
            
            if img_array is None:
                return JsonResponse({'error': 'Failed to preprocess image'}, status=400)
            
            # Make prediction
            pred = model.predict(img_array, verbose=0)
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
