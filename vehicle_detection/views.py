import sys

sys.path.insert(0, "./")
sys.path.insert(0, "../")

from django.http import StreamingHttpResponse
from config_helper import ReadConfigFile
from vehicle_detection.frame_grabber import *
from vehicle_detection.vehicle_detection import *

from loguru import logger

config = ReadConfigFile().read_config("config.json")
frame_grabber = RTSPframeGrabber(config.camera_url)
yolo = YoloLpd()
counter = VehicleCounter()


def cam(camera):
    while True:
        try:
            frame_grabber.start()
            frame = camera.latest_frame()
            _, jpeg = cv2.imencode(".jpg", frame)
            byte_img = jpeg.tobytes()
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + byte_img + b"\r\n\r\n")
        except Exception as error:
            logger.error(f"[CAM] Frame not received [error={error}]")


def video_feed():
    while True:
        frame_grabber.start()
        img = frame_grabber.latest_frame()
        boxes, confs, class_ids = yolo.predict_lpd(img)

        img_labeled = yolo.draw_boxes(img, boxes, confs, class_ids,
                                      yolo.get_labels())
        flag, img_enc = cv2.imencode(".jpg", img_labeled)

        if not flag:
            continue
        img_byte = img_enc.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + img_byte + b'\r\n\r\n')


def vehicle_counter():

    previous_frame_detections = [{
        (0, 0): 0
    } for i in range(counter.FRAMES_BEFORE_CURRENT)]

    vehicle_count = 0

    while True:
        boxes, confidences, classIDs = [], [], []
        vehicle_crossed_line_flag = config.counter_bbox

        frame_grabber.start()
        img = frame_grabber.latest_frame()
        if not frame_grabber.grabbed:
            break

        boxes, confidences, classIDs = yolo.predict_lpd(img)

        vehicle_count, current_detections = counter.count_vehicles(
            boxes, classIDs, vehicle_count, previous_frame_detections, img)

        counter.display_vehicle_count(img, vehicle_count)
        logger.info(f"[COUNTER] the number of counted vehicles {vehicle_count}")

        # Updating with the current frame detections
        #Removing the first frame from the list
        previous_frame_detections.pop(0)
        # previous_frame_detections.append(spatial.KDTree(current_detections))
        previous_frame_detections.append(current_detections)

        # Draw detection box
        # counter.draw_detection_boxes(boxes, classIDs, confidences, img)
        yolo.draw_boxes(img, boxes, confidences, classIDs, yolo.get_labels())

        flag, img_enc = cv2.imencode(".jpg", img)

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


def render_vehicle_counter_video(request):
    return StreamingHttpResponse(
        vehicle_counter(),
        content_type="multipart/x-mixed-replace; boundary=frame")
