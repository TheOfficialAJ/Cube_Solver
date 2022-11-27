import cv2
import cv2 as cv
import numpy as np


class Calibrator:
    def __init__(self):
        self.vidstream = cv.VideoCapture(0)
        self.key = None
        cv2.namedWindow("Display", cv2.WINDOW_AUTOSIZE)

    def updateFrame(self):
        ret_val, frame = self.vidstream.read()
        cv.imshow("Display", frame)
        self.key = cv.waitKey(20)

    def exit(self):
        cv.destroyAllWindows()


if __name__ == '__main__':
    calibrator = Calibrator()
    while True:
        calibrator.updateFrame()
        if calibrator.key == 27:  # Terminate on pressing Esc
            break
        if cv2.getWindowProperty("Display", cv2.WND_PROP_VISIBLE) < 1:  # Terminate on pressing the red X
            break
    calibrator.exit()
