from django.urls import path
from . import views

app_name = 'classifier'  # Add namespace for better URL organization

urlpatterns = [
    path('', views.predict_flower, name='home'),  # classifier/ goes directly to predict
    path('predict/', views.predict_flower, name='predict'),  # Keep original path for compatibility
]
