from django.contrib import admin
from django.urls import path
from myWeb import views

urlpatterns = [
    path('', views.welcome),
    path('welcome/', views.welcome),
    path('index/', views.index),
    path('login/', views.login),
    path('register/', views.register),
    path('logout/', views.logout),
    path('index_use/', views.index_use),
    path('upload/', views.upload),
    path('start_training/', views.start_training),
    path('index_to_progress/', views.index_to_progress),
    path('visualization/', views.visualization),
]