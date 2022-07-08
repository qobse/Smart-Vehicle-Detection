import sys

sys.path.insert(0, "./")
sys.path.insert(0, "../")

from django.http import StreamingHttpResponse
from config_helper import ReadConfigFile
from vehicle_detection.frame_grabber import *
from vehicle_detection.vehicle_detection import *

from loguru import logger

config = ReadConfigFile().read_config("config.json")
frame_grabber = RTSPframeGrabber(config.camera_url).start()
yolo = YoloLpd()


def cam(camera):
    while True:
        try:
            frame = camera.latest_frame()
            _, jpeg = cv2.imencode(".jpg", frame)
            byte_img = jpeg.tobytes()
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + byte_img + b"\r\n\r\n")
        except AttributeError:
            logger.info("[INFO] Frame grabber failed")


def video_feed():
    while True:
        img = frame_grabber.latest_frame()
        boxes, confs, class_ids = yolo.predict_lpd(img)

        img_labeled = yolo.draw_boxes(img, boxes, confs, class_ids,
                                      yolo.get_labels(config.lpd_classes))
        flag, img_enc = cv2.imencode(".jpg", img_labeled)

        if not flag:
            continue
        img_byte = img_enc.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + img_byte + b'\r\n\r\n')


def render_camera_stream(request):
    return StreamingHttpResponse(
        cam(frame_grabber),
        content_type="multipart/x-mixed-replace; boundary=frame")


def render_detection_video(request):
    return StreamingHttpResponse(
        video_feed(), content_type="multipart/x-mixed-replace; boundary=frame")
