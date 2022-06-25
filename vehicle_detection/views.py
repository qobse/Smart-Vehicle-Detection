import sys

sys.path.insert(0, "./")
sys.path.insert(0, "../")

from django.http import StreamingHttpResponse
from config_helper import ReadConfigFile
from vehicle_detection.frame_grabber import *

from loguru import logger

config = ReadConfigFile().read_config("config.json")
frame_grabber = RTSPframeGrabber(config.camera_url)


def cam(camera):
    while True:
        try:
            frame = camera.latest_frame()
            _, jpeg = cv2.imencode(".jpg", frame)
            byte_img = jpeg.tobytes()
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + byte_img +
                   b"\r\n\r\n")
        except AttributeError:
            logger.info("[INFO] Frame grabber failed")


def render_camera_stream(request):
    return StreamingHttpResponse(
        cam(frame_grabber),
        content_type="multipart/x-mixed-replace; boundary=frame")
