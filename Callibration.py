import cv2 as cv
import cuberesolver
import numpy as np

x1 = x2 = x3 = x4 = None

vidstream = cv.VideoCapture(0)


def getAverageColor(img_ROI):  # Returns the average color of ROI in LAB color space
    l = []
    a = []
    b = []
    img_ROI = cv.cvtColor(img_ROI, cv.COLOR_BGR2Lab)
    avg_color_per_row = np.average(img_ROI, axis=0)
    averageCol = np.average(avg_color_per_row, axis=0)

    return averageCol


def getROI(event, x, y, flags, param):
    global x1, x2, y1, y2
    print("Event registered")
    if event == cv.EVENT_LBUTTONDOWN:
        x1, y1 = x, y

    if event == cv.EVENT_LBUTTONUP:
        x2, y2 = x, y
        img_roi = frame[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)]
        cv.rectangle(modImg, (x1, y1), (x2, y2), (0, 255, 0), 1)
        print(getAverageColor(img_roi))


cv.namedWindow('Webcam')
cv.setMouseCallback("Webcam", getROI)

while True:
    ret, frame = vidstream.read()
    modImg = np.copy(frame)
    cv.imshow("Webcam", modImg)
    k = cv.waitKey(20)

    if k == 27:
        break
