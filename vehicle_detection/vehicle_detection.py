######################## Import libraries ########################
import sys

from django import conf

sys.path.insert(0, "./")
sys.path.insert(0, "../")
from config_helper import ReadConfigFile
from vehicle_detection.frame_grabber import *
from vehicle_detection.image_helper import *
from loguru import logger
import numpy as np

import cv2
import os
from pathlib import Path

sys.path.insert(0, "../")
BASE_DIR = Path(__file__).resolve().parent.parent


class YoloLpd():

    def __init__(self, path="config.json") -> None:
        self.config = ReadConfigFile().read_config(path)
        self.lpd_net = self.init_model()

    def init_model(self):
        lpd_net = cv2.dnn.readNet(
            str(BASE_DIR) + "/" + self.config.lpd_network,
            str(BASE_DIR) + "/" + self.config.lpd_weights)
        lpd_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
        lpd_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        dummy_img = np.zeros((self.config.lpd_height, self.config.lpd_width, 3),
                             np.uint8)
        dummy_blob = cv2.dnn.blobFromImage(dummy_img)

        lpd_net.setInput(dummy_blob)
        lpd_net.forward()
        logger.info("[LPD] YOLO DNN model initialization completed ...")

        return lpd_net

    def get_output_layers(self, net):
        return net.getLayerNames()

    def get_labels(self):
        labels = open(self.config.lpd_classes).read().strip().split("\n")
        return labels

    def pre_processing(self, img):
        org_img, img = readImage_OriResize(img, (416, 416),
                                           gray=False,
                                           resizeMode=cv2.INTER_LINEAR)
        (resized_height, resized_width) = img.shape[:2]
        blob = cv2.dnn.blobFromImage(img,
                                     1 / 255, (resized_height, resized_width),
                                     swapRB=True,
                                     crop=False)
        return blob

    def clip_coords(self, input_value, min_value, max_value):
        return min(max(input_value, min_value), max_value)

    def nms(self, box_list, conf_list, class_id_list):
        boxes = []
        confs = []
        class_ids = []

        idxs = cv2.dnn.NMSBoxes(box_list, conf_list, self.config.lpd_confidence,
                                self.config.lpd_nms_threshold)
        for i in range(len(box_list)):
            for i in idxs:
                boxes.append(box_list[int(i)])
                confs.append(conf_list[int(i)])
                class_ids.append(class_id_list[int(i)])
        return boxes, confs, class_ids

    def draw_boxes(self,
                   img,
                   boxes,
                   confs,
                   class_ids,
                   labels,
                   adjusting_coor=[0, 0]):
        #returns labeled numpy image
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
            cv2.putText(img, label + " " + str(format(conf, ".2f")), (x, y - 5),
                        font, 4, color, 4)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 5)
        return img

    def crop_img(self, img, box):
        #bbox[x_min, y_min, width, height]
        x, y, w, h = box
        cropped_img = img[y:y + h, x:x + w]
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
            inf_time = (time.time() - t1)
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

            post_proc_time = (time.time() - t2)
            logger.info(f"[INFO] YOLO Inference time: {inf_time}s")
            logger.info(f"[INFO] YOLO PostProcessing time: {post_proc_time}s")

            boxes, confs, class_ids = self.nms(boxes, confs, class_ids)
            return boxes, confs, class_ids
