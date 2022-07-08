from django.urls import path
from . import views

urlpatterns = [
    path("camera/", views.render_camera_stream, name="camera"),
    path("lpd/", views.render_detection_video, name="lpd"),
]
