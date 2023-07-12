import time
from threading import Lock
from threading import Thread

import cv2
from loguru import logger


class RTSPframeGrabber(object):
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)

        # Take screenshot every x seconds
        self.screenshot_interval = 1

        # Default resolutions of the frame are obtained (system dependent)
        self.frame_width = int(self.stream.get(3))
        self.frame_height = int(self.stream.get(4))
        self.fps = int(self.stream.get(cv2.CAP_PROP_FPS))

        (self.grabbed, self.frame) = self.stream.read()
        self.started = False
        self.read_lock = Lock()

    def start(self):
        try:
            if self.started:
                return None
            self.started = True
            # Start the thread to read frames from the video stream
            self.thread = Thread(target=self.update, args=())
            # self.thread.daemon = True
            self.thread.start()
            logger.info(f"[CAM] Frame grabber initialization completed")
            return self
        except Exception as error:
            logger.error(f"[CAM] Frame grabber initialization failed [error={error}]")

    def update(self):
        # Read the next frame from the stream in a different thread
        while self.started:
            (grabbed, frame) = self.stream.read()
            self.read_lock.acquire()
            self.grabbed = grabbed
            self.frame = frame
            self.read_lock.release()

    def latest_frame(self):
        self.read_lock.acquire()
        frame = self.frame.copy()
        self.read_lock.release()
        return frame

    def stop(self):
        self.started = False
        self.thread.join()

    def save_frame(self):
        # Save obtained frame periodically
        self.frame_count = 0

        def save_frame_thread():
            while True:
                try:
                    cv2.imwrite("frame_{}.png".format(self.frame_count), self.frame)
                    self.frame_count += 1
                    time.sleep(self.screenshot_interval)
                except AttributeError:
                    pass

        Thread(target=save_frame_thread, args=()).start()

    def __exit__(self, exc_type, exc_value, traceback):
        self.stream.release()
