from django.urls import path
from . import views

urlpatterns = [
    path('predict/', views.predict_flower, name='predict'),
]
