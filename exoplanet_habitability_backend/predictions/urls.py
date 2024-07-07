from django.urls import path
from . import views

urlpatterns = [
    path('predict/', views.predict_habitability, name='predict_habitability'),
]
