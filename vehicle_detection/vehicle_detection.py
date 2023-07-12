######################## Import libraries ########################
import sys

from django import conf

sys.path.insert(0, "./")
sys.path.insert(0, "../")
from config_helper import ReadConfigFile
from vehicle_detection.image_helper import *
from loguru import logger
from scipy import spatial
import numpy as np

import cv2
import os
from pathlib import Path

sys.path.insert(0, "../")
BASE_DIR = Path(__file__).resolve().parent.parent


class YoloLpd:
    def __init__(self, path="config.json") -> None:
        self.config = ReadConfigFile().read_config(path)
        self.lpd_net = self.init_model()

    def init_model(self):
        lpd_net = cv2.dnn.readNet(
            str(BASE_DIR) + "/" + self.config.lpd_network,
            str(BASE_DIR) + "/" + self.config.lpd_weights,
        )
        lpd_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
        lpd_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        dummy_img = np.zeros(
            (self.config.lpd_height, self.config.lpd_width, 3), np.uint8
        )
        dummy_blob = cv2.dnn.blobFromImage(dummy_img)

        lpd_net.setInput(dummy_blob)
        lpd_net.forward()
        logger.info("[LPD] YOLO DNN model initialization completed")

        return lpd_net

    def get_output_layers(self, net):
        return net.getLayerNames()

    def get_labels(self):
        labels = open(self.config.lpd_classes).read().strip().split("\n")
        return labels

    def pre_processing(self, img):
        org_img, img = readImage_OriResize(
            img, (416, 416), gray=False, resizeMode=cv2.INTER_LINEAR
        )
        (resized_height, resized_width) = img.shape[:2]
        blob = cv2.dnn.blobFromImage(
            img, 1 / 255, (resized_height, resized_width), swapRB=True, crop=False
        )
        return blob

    def clip_coords(self, input_value, min_value, max_value):
        return min(max(input_value, min_value), max_value)

    def nms(self, box_list, conf_list, class_id_list):
        boxes = []
        confs = []
        class_ids = []

        idxs = cv2.dnn.NMSBoxes(
            box_list,
            conf_list,
            self.config.lpd_confidence,
            self.config.lpd_nms_threshold,
        )
        for i in range(len(box_list)):
            for i in idxs:
                boxes.append(box_list[int(i)])
                confs.append(conf_list[int(i)])
                class_ids.append(class_id_list[int(i)])
        return boxes, confs, class_ids

    def draw_boxes(self, img, boxes, confs, class_ids, labels, adjusting_coor=[0, 0]):
        # returns labeled numpy image
        # labels = ["MyLP", "PhLP", "person", "bicycle", "car", "motorcycle", "bus", "truck"]
        colors = np.random.uniform(0, 255, size=(100, 3))
        font = cv2.FONT_HERSHEY_PLAIN
        for i, box in enumerate(boxes):
            x, y, w, h = box
            x = x + adjusting_coor[0]
            y = y + adjusting_coor[1]
            label = str(labels[class_ids[i]])
            conf = confs[i]
            color = colors[i]
            cv2.putText(
                img,
                label + " " + str(format(conf, ".2f")),
                (x, y - 5),
                font,
                2,
                color,
                2,
            )
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        return img

    def draw_ROI(self, img):
        colors = (100, 50, 200)
        cv2.polylines(
            img,
            [np.array([(2550, 450), (2200, 0), (0, 1400), (2000, 1400)], np.int32)],
            True,
            (15, 220, 10),
            6,
        )
        return img

    def crop_img(self, img, box):
        # bbox[x_min, y_min, width, height]
        x, y, w, h = box
        cropped_img = img[y : y + h, x : x + w]
        return cropped_img

    def predict_lpd(self, img):
        while True:
            ################################## initialize parameters ###################################
            confs = []
            boxes = []
            class_ids = []

            ################################## load img ##############################################
            # img = IpCam(self.url).get_img()
            if isinstance(img, np.ndarray):
                img_org = img
            else:
                img_org = np.asarray(bytearray(img), dtype="uint8")
                img_org = cv2.imdecode(img_org, cv2.IMREAD_COLOR)
            img_org_shape = img_org.shape

            ################################## preprocess & inference ##################################
            ln = self.get_output_layers(self.lpd_net)

            blob = self.pre_processing(img_org)
            t1 = time.time()
            self.lpd_net.setInput(blob)
            inf_time = time.time() - t1
            (detect_boxes, detect_clas) = self.lpd_net.forward(ln)

            ################################## post peocess the model output ###########################
            t2 = time.time()
            for i, detection in enumerate(detect_boxes):
                scores = detect_clas[i]
                class_id = np.argmax(scores)
                conf = float(scores[class_id])
                if conf > self.config.lpd_confidence:
                    # yapf: disable
                    x_center    = int(detection[0] * img_org_shape[1])
                    y_center    = int(detection[1] * img_org_shape[0])
                    width       = int(detection[2] * img_org_shape[1])
                    height      = int(detection[3] * img_org_shape[0])

                    x_min       = int(x_center - (width / 2))
                    y_min       = int(y_center - (height / 2))

                    x_min       = self.clip_coords(x_min, 0, img_org_shape[1])
                    y_min       = self.clip_coords(y_min, 0, img_org_shape[0])
                    # yapf: enable

                    boxes.append([x_min, y_min, width, height])
                    confs.append(float(conf))
                    class_ids.append(class_id)

            post_proc_time = time.time() - t2
            # logger.info(f"[INFO] YOLO Inference time: {inf_time}s")
            # logger.info(f"[INFO] YOLO PostProcessing time: {post_proc_time}s")

            boxes, confs, class_ids = self.nms(boxes, confs, class_ids)
            return boxes, confs, class_ids


class VehicleCounter(YoloLpd):
    def __init__(self) -> None:
        super().__init__()

        self.FRAMES_BEFORE_CURRENT = 10
        self.LABELS = YoloLpd().get_labels()

    def display_vehicle_count(self, frame, vehicle_count):
        cv2.putText(
            frame,  # Image
            "Detected Vehicles: " + str(vehicle_count),  # Label
            (20, 20),  # Position
            cv2.FONT_HERSHEY_SIMPLEX,  # Font
            0.8,  # Size
            (0, 0xFF, 0),  # Color
            2,  # Thickness
            cv2.FONT_HERSHEY_COMPLEX_SMALL,
        )

    def box_and_line_overlap(self, x_mid_point, y_mid_point, line_coordinates):
        x1_line, y1_line, x2_line, y2_line = line_coordinates  # Unpacking

        if (x_mid_point >= x1_line and x_mid_point <= x2_line + 5) and (
            y_mid_point >= y1_line and y_mid_point <= y2_line + 5
        ):
            return True
        return False

    def display_fps(self, start_time, num_frames):
        current_time = int(time.time())
        if current_time > start_time:
            os.system("clear")  # Equivalent of CTRL+L on the terminal
            print("FPS:", num_frames)
            num_frames = 0
            start_time = current_time
        return start_time, num_frames

    def draw_detection_boxes(self, boxes, classIDs, confidences, frame):
        np.random.seed(42)
        COLORS = np.random.randint(0, 255, size=(len(self.LABELS), 3), dtype="uint8")
        # ensure at least one detection exists
        # if len(idxs) > 0:
        # loop over the indices we are keeping
        for i, box in enumerate(boxes):
            # extract the bounding box coordinates
            (x, y) = (box[0], box[1])
            (w, h) = (box[2], box[3])

            # draw a bounding box rectangle and label on the frame
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(self.LABELS[classIDs[i]], confidences[i])
            cv2.putText(
                frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )
            # Draw a green dot in the middle of the box
            cv2.circle(
                frame, (x + (w // 2), y + (h // 2)), 2, (0, 0xFF, 0), thickness=2
            )

    def initialize_video_writer(
        self,
        video_width,
        video_height,
        fps,
        save_path,
    ):
        # Getting the fps of the source video
        sourceVideofps = fps
        # initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        return cv2.VideoWriter(
            save_path, fourcc, sourceVideofps, (video_width, video_height), True
        )

    def box_in_previous_frames(
        self, previous_frame_detections, current_box, current_detections
    ):
        centerX, centerY, width, height = current_box
        dist = np.inf  # Initializing the minimum distance
        # Iterating through all the k-dimensional trees
        for i in range(self.FRAMES_BEFORE_CURRENT):
            coordinate_list = list(previous_frame_detections[i].keys())
            if (
                len(coordinate_list) == 0
            ):  # When there are no detections in the previous frame
                continue
            # Finding the distance to the closest point and the index
            temp_dist, index = spatial.KDTree(coordinate_list).query(
                [(centerX, centerY)]
            )
            if temp_dist < dist:
                dist = temp_dist
                frame_num = i
                coord = coordinate_list[index[0]]

        if dist > (max(width, height) / 2):
            return False

        # Keeping the vehicle ID constant
        current_detections[(centerX, centerY)] = previous_frame_detections[frame_num][
            coord
        ]
        return True

    def count_vehicles(
        self, boxes, classIDs, vehicle_count, previous_frame_detections, frame
    ):
        list_of_vehicles = ["bicycle", "car", "motorbike", "bus", "truck", "train"]
        current_detections = {}
        # ensure at least one detection exists
        # if len(idxs) > 0:
        # loop over the indices we are keeping
        for i, box in enumerate(boxes):
            # extract the bounding box coordinates
            (x, y) = (box[0], box[1])
            (w, h) = (box[2], box[3])

            centerX = x + (w // 2)
            centerY = y + (h // 2)

            # When the detection is in the list of vehicles, AND
            # it crosses the line AND
            # the ID of the detection is not present in the vehicles
            if self.LABELS[classIDs[i]] in list_of_vehicles:
                current_detections[(centerX, centerY)] = vehicle_count
                if not self.box_in_previous_frames(
                    previous_frame_detections,
                    (centerX, centerY, w, h),
                    current_detections,
                ):
                    vehicle_count += 1
                    # vehicle_crossed_line_flag += True
                # else: #ID assigning
                # Add the current detection mid-point of box to the list of detected items
                # Get the ID corresponding to the current detection

                ID = current_detections.get((centerX, centerY))
                # If there are two detections having the same ID due to being too close,
                # then assign a new ID to current detection.
                if list(current_detections.values()).count(ID) > 1:
                    current_detections[(centerX, centerY)] = vehicle_count
                    vehicle_count += 1

                # Display the ID at the center of the box
                cv2.putText(
                    frame,
                    str(ID),
                    (centerX, centerY),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    [0, 0, 255],
                    2,
                )

        return vehicle_count, current_detections
