from django.urls import path
from . import views

app_name = 'classifier'  # Add namespace for better URL organization

urlpatterns = [
    path('', views.predict_flower, name='home'),  # classifier/ goes directly to predict
    path('predict/', views.predict_flower, name='predict'),  # Keep original path for compatibility
    path('model-status/', views.model_status, name='model_status'),  # New: AJAX endpoint for model loading status
    path('api/predict/', views.api_predict, name='api_predict'),
    path('test/', views.test_model, name='test_model'),# Optional: API endpoint
]
