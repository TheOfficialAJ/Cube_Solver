import cv2 as cv
import numpy as np

# Uses LAB colorspace to reduce the effect of lighting conditions on color detection
# It makes masks for all colors and then choose the color for which the mask gives the brightest values
class ColorDetector:
    def __init__(self):
        """ NOTE: White color is the default color so any square whose color is not one of the other 5 is assigned white color """
        self.image = None
        self.COLORS = {'BLUE': (240, 250, 3), 'RED': (0, 0, 255), 'GREEN': (0, 255, 0), 'YELLOW': (0, 255, 255),
                       'ORANGE': (0, 128, 255), 'WHITE': (255, 255,
                                                          255)}

        self.COLOR_RANGES = {'YELLOW': [(20, 110, 140), (255, 130, 180)], 'ORANGE': [(100, 155, 150), (230, 180, 175)],
                             'RED': [(30, 140, 120), (140, 160, 155)], 'GREEN': [(40, 80, 120), (220, 110, 150)],
                             'BLUE': [(20, 95, 75), (220, 125, 115)]}
        cv.namedWindow("Color Detector", cv.WINDOW_NORMAL)
        cv.setMouseCallback("Color Detector", self.onClick)
        self.squares = []

    def predictColor(self, image, contour):
        self.image = image
        cv.imshow("Color Detector", self.image)
        roi = self.cropMinAreaRect(image, contour)
        roi = cv.cvtColor(roi, cv.COLOR_BGR2Lab)
        # cv.imshow("ROI", roi)
        # print(roi.shape)
        color_masks = {}
        for color_name, color_range in self.COLOR_RANGES.items():
            color_masks[color_name] = cv.inRange(roi, color_range[0], color_range[1])

        masked_images = {}
        for color_name, mask in color_masks.items():
            masked_images[color_name] = cv.threshold(cv.bitwise_and(roi, roi, mask=mask), 0, 255, cv.THRESH_BINARY)[1]

        prediction = "WHITE"
        maxVal = 0
        for color_name, roi_mask in masked_images.items():
            average_col = self.getAverageColor(roi_mask)[0]
            print(color_name, average_col)
            if average_col > maxVal:  # If this mask is more whiter than any previous one, change prediction to this color
                prediction = color_name
                maxVal = average_col

        # for col_name, img in masked_images.items():
        #     cv.imshow(col_name + " MASK", img)
        return prediction

    def onClick(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            img_lab = cv.cvtColor(self.image, cv.COLOR_BGR2Lab)
            print(img_lab[y][x])

    def getSquareColor(self, image,
                       contour):  # Returns the contour image drawn with the color of the contour and the color name
        color = self.predictColor(image, contour)
        cv.drawContours(image, [contour], 0, self.COLORS[color], 2)
        cv.imshow("Color Detector", image)
        return image, color

    def cropMinAreaRect(self, img, cnt):
        rect = cv.minAreaRect(cnt)
        box = cv.boxPoints(rect)
        box = np.int0(box)

        W = rect[1][0]
        H = rect[1][1]

        Xs = [i[0] for i in box]
        Ys = [i[1] for i in box]
        x1 = min(Xs)
        x2 = max(Xs)
        y1 = min(Ys)
        y2 = max(Ys)

        angle = rect[2]
        if angle < -45:
            angle += 90

        # Center of rectangle in source image
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        # Size of the upright rectangle bounding the rotated rectangle
        size = (x2 - x1, y2 - y1)
        M = cv.getRotationMatrix2D((size[0] / 2, size[1] / 2), angle, 1.0)
        # Cropped upright rectangle
        cropped = cv.getRectSubPix(img, size, center)
        cropped = cv.warpAffine(cropped, M, size)
        croppedW = H if H > W else W
        croppedH = H if H < W else W
        # Final cropped & rotated rectangle
        croppedRotated = cv.getRectSubPix(cropped, (int(croppedW), int(croppedH)), (size[0] / 2, size[1] / 2))
        return croppedRotated

    def getAverageColor(self, img_ROI):  # Returns the average color of ROI in LAB color space
        l = []
        a = []
        b = []
        img_ROI = cv.cvtColor(img_ROI, cv.COLOR_BGR2Lab)
        # for x in range(img_ROI.shape[1]):
        #     for y in range(img_ROI.shape[0]):
        #         l.append((img_ROI[y][x])[0])
        #         a.append((img_ROI[y][x])[1])
        #         b.append((img_ROI[y][x])[2])
        avg_color_per_row = np.average(img_ROI, axis=0)
        averageCol = np.average(avg_color_per_row, axis=0)

        # averageCol = round(np.mean(l)), round(np.mean(a)), round(np.mean(b))
        return averageCol


if __name__ == '__main__':
    while True:
        colorDetector = ColorDetector()
