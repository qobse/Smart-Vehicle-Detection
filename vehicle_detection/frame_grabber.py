from threading import Thread
import time
import cv2


class RTSPframeGrabber(object):

    def __init__(self, src=0):
        # Create a VideoCapture object
        self.capture = cv2.VideoCapture(src)

        # Take screenshot every x seconds
        self.screenshot_interval = 1

        # Default resolutions of the frame are obtained (system dependent)
        self.frame_width = int(self.capture.get(3))
        self.frame_height = int(self.capture.get(4))

        # Start the thread to read frames from the video stream
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        # Read the next frame from the stream in a different thread
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()
            time.sleep(0.01)

    def latest_frame(self):
        return self.frame

    def show_frame(self):
        # Display frames in main program
        if self.status:
            cv2.imshow("frame", self.frame)

        # Press Q on keyboard to stop recording
        key = cv2.waitKey(1)
        if key == ord("q"):
            self.capture.release()
            cv2.destroyAllWindows()
            exit(1)

    def save_frame(self):
        # Save obtained frame periodically
        self.frame_count = 0

        def save_frame_thread():
            while True:
                try:
                    cv2.imwrite("frame_{}.png".format(self.frame_count),
                                self.frame)
                    self.frame_count += 1
                    time.sleep(self.screenshot_interval)
                except AttributeError:
                    pass

        Thread(target=save_frame_thread, args=()).start()