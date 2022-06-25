from django.urls import path
from . import views

urlpatterns = [
    path("camera/", views.render_camera_stream, name="camera"),
]
