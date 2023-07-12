import time

import cv2
import numpy as np


def image_getChannel(img):
    shape = img.shape
    if len(shape) == 2:
        return 1
    elif len(shape) == 3:
        return shape[2]
    else:
        assert "Error Reading Image Channel from shape (%d)" % (
            ",".join(map(str, shape))
        )


def readImage(imgPath, gray=False):
    if type(imgPath) is np.ndarray:
        if gray:
            return image_toGray(imgPath)
        else:
            return imgPath
    img = cv2.imread(imgPath, 0 if gray else 1)
    if img is None:
        raise Exception("Invalid Image Path")
    return img


def readImage_OriResize(imgPath, targetSize, gray=False, resizeMode=cv2.INTER_LINEAR):
    """
    return oriSize_img, resizedImg
    """
    original_image = readImage(imgPath, gray)
    resized_image = cv2.resize(original_image, targetSize, interpolation=resizeMode)
    return original_image, resized_image


def image_toGray(imgPath):
    img = readImage(imgPath)
    c = image_getChannel(img)
    if c == 1:
        # print("image_toGray: Channel is already 1")
        return img
    elif c == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        assert "Invalid Channel Count: not 1 or 3"
