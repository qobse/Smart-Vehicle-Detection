from django.urls import path
from . import views

urlpatterns = [
    path("camera/", views.render_camera_stream, name="camera"),
    path("lpd/", views.render_detection_video, name="lpd"),
    path("counter/", views.render_vehicle_counter_video, name="counter"),
]
