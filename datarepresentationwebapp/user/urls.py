from django.urls import path

from . import views

urlpatterns = [
    path('login', views.LoginView.as_view(), name='login'),
    path('register', views.UserCreateView.as_view(), name='register'),
    path('logout', views.LogoutView.as_view(), name="logout")
]