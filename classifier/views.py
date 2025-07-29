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
                        # Try to load and use the model
                        try:
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
                                    raise FileNotFoundError("Model file not found")
                            
                            print(f"Loading model from: {MODEL_PATH}")
                            
                            # Load model (this happens only when making a prediction)
                            model = load_model(MODEL_PATH, compile=False)
                            
                            # Reset file pointer for processing
                            uploaded_file.seek(0)
                            
                            # Load and resize the image to match model's input shape (180x180)
                            img = Image.open(uploaded_file).convert('RGB').resize((180, 180))

                            # Convert image to numpy array and normalize
                            img_array = np.array(img) / 255.0

                            # Add batch dimension
                            img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 180, 180, 3)

                            # Make prediction
                            pred = model.predict(img_array, verbose=0)
                            predicted_class = CLASS_NAMES[np.argmax(pred)]
                            confidence = float(np.max(pred))
                            confidence_score = confidence

                            # Set prediction message with confidence
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
    
    try:
        form = UploadImageForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = request.FILES['image']
            
            # Load model
            from tensorflow.keras.models import load_model
            MODEL_PATH = os.path.join(settings.BASE_DIR, 'classifier', 'flower_model (1).keras')
            
            if not os.path.exists(MODEL_PATH):
                return JsonResponse({'error': 'Model file not found'}, status=503)
            
            model = load_model(MODEL_PATH, compile=False)
            
            # Process image
            img = Image.open(uploaded_file).convert('RGB').resize((180, 180))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
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
