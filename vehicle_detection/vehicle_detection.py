######################## Import libraries ########################
import sys

sys.path.insert(0, "./")
sys.path.insert(0, "../")
from config_helper import ReadConfigFile
from vehicle_detection.frame_grabber import *
from vehicle_detection.image_helper import *
from loguru import logger
import signal
import numpy as np

import cv2
import os

lbl = [
    "MyLP", "PhLP", "person", "bicycle", "car", "motorcycle", "bus", "truck"
]


def signal_handler(signal, frame):
    logger.info(f"[SIG] Caught signal {signal}")

    is_running = False


######################## Initialize parameters and DNN ###########
is_running = True

config = ReadConfigFile().read_config("config.json")
frame_grabber = RTSPframeGrabber(config.camera_url).start()

frame = frame_grabber.latest_frame()

lpd_net = cv2.dnn.readNet(config.lpd_network, config.lpd_weights)
lpd_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
lpd_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

ln = lpd_net.getLayerNames()
######################## Set signal handler ######################
labels = open(config.lpd_classes).read().strip().split("\n")
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

while is_running:
    org_img, img = readImage_OriResize(frame, (416, 416),
                                       gray=False,
                                       resizeMode=cv2.INTER_LINEAR)

    (org_height, org_width) = org_img.shape[:2]
    (resized_height, resized_width) = img.shape[:2]

    blob = cv2.dnn.blobFromImage(img,
                                 1 / 255, (resized_height, resized_width),
                                 swapRB=True,
                                 crop=False)

    lpd_net.setInput(blob)
    start = time.time()
    (detect_boxes, detect_clas) = lpd_net.forward(ln)
    end = time.time()

    boxes = []
    confidences = []
    classIDs = []
    for i, detection in enumerate(detect_boxes):
        scores = detect_clas[i]
        classID = np.argmax(scores)
        confidence = float(scores[classID])
        if confidence > config.lpd_confidence:
            box = detection * np.array(
                [org_width, org_height, org_width, org_height])
            (x_center, y_center, width, height) = box.astype("int")
            x_min = int(x_center - (width / 2))
            y_min = int(y_center - (height / 2))
            boxes.append([x_min, y_min, int(width), int(height)])
            confidences.append(float(confidence))
            classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, config.lpd_confidence,
                            config.lpd_nms_threshold)

    # ensure at least one detection exists
    if len(idxs) > 0:
        for i in idxs.flatten():
            if classIDs[i] <= 9:
                (x_min, y_min) = (boxes[i][0], boxes[i][1])
                (width, height) = (boxes[i][2], boxes[i][3])
                lp = org_img[y_min:y_min + height, x_min:x_min + width]
                cv2.imwrite(
                    os.path.join(
                        "/home/yaqoob/work/others/svd/vehicle_detection/lp",
                        str(i) + ".jpg",
                    ),
                    lp,
                )
                # draw a bounding box rectangle and label on the image
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(
                    org_img,
                    (x_min, y_min),
                    (x_min + width, y_min + height),
                    color,
                    2,
                )
                text = "{}: {:.4f}".format(labels[classIDs[i]], confidences[i])
                cv2.putText(
                    org_img,
                    text,
                    (x_min, y_min - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )
                cv2.imwrite(
                    os.path.join(
                        "/home/yaqoob/work/others/svd/vehicle_detection/output",
                        str(i) + ".jpg",
                    ),
                    frame,
                )
######################## Preprocessing ###########################
