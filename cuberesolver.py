import cv2 as cv
import numpy as np
import colour as clr
from resize import resize


class CubeResolver:
    def __init__(self, image, SCALE):
        self.image = image
        self.SCALE = SCALE

    def getDiagonalCorners(self, contour_corners):
        x1 = int(contour_corners[0][0])
        x2 = int(contour_corners[1][0])
        x3 = int(contour_corners[2][0])
        x4 = int(contour_corners[3][0])
        y1 = int(contour_corners[0][1])
        y2 = int(contour_corners[1][1])
        y3 = int(contour_corners[2][1])
        y4 = int(contour_corners[3][1])
        top_left_x = min([x1, x2, x3, x4])
        top_left_y = min([y1, y2, y3, y4])
        bot_right_x = max([x1, x2, x3, x4])
        bot_right_y = max([y1, y2, y3, y4])
        return (top_left_x, top_left_y), (bot_right_x, bot_right_y)


        # def getClosestColor(test_img):
        #     print(test_img)
        #     best_fit = 100
        #     test_img_lab = np.array(cv.cvtColor(test_img, cv.COLOR_BGR2Lab))
        #     for color_name, color in COLORS.items():
        #         color = np.array([color], dtype='uint8')
        #         color_lab = cv.cvtColor(color.astype(np.float32) / 255, cv.COLOR_BGR2Lab)
        #         # color = [color]  # To convert BGR value to item in list
        #         # color = np.float32(color)
        #         # color_lab = cv.cvtColor(color, cv.COLOR_BGR2Lab
        #         print("color", color_lab)
        #         delta_e = clr.delta_E(test_img_lab, color_lab, method='cie 2000')
        #         delta_e = np.median(delta_e)
        #         print("DELTA E for", color_name, delta_e)
        #         if delta_e < best_fit:
        #             best_fit = delta_e
        #             predicted_color = color_name
        #     print("Color is ", predicted_color)

    def distance(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        return dist

    def getAverageColor(self, img_ROI):  # Returns the average color of ROI in LAB color space
        l = []
        a = []
        b = []
        img_ROI = cv.cvtColor(img_ROI, cv.COLOR_BGR2Lab)
        for x in range(img_ROI.shape[1]):
            for y in range(img_ROI.shape[0]):
                l.append((img_ROI[y][x])[0])
                a.append((img_ROI[y][x])[1])
                b.append((img_ROI[y][x])[2])

        return round(np.mean(l)), round(np.mean(a)), round(np.mean(b))

    def predictColor(self, image, corners, color_ranges):
        diagCorners = getDiagonalCorners(corners)
        image_lab = cv.cvtColor(image, cv.COLOR_BGR2Lab)
        roi = image_lab[diagCorners[0][1]:diagCorners[1][1], diagCorners[0][0]:diagCorners[1][0]]

        color_masks = {}
        for color_name, color_range in color_ranges.items():
            color_masks[color_name] = cv.inRange(roi, color_range[0], color_range[1])

        masked_images = {}
        for color_name, mask in color_masks.items():
            masked_images[color_name] = cv.threshold(cv.bitwise_and(roi, roi, mask=mask), 0, 255, cv.THRESH_BINARY)[1]

        prediction = None
        maxVal = 0
        for color_name, roi_mask in masked_images.items():
            average_col = getAverageColor(masked_images[color_name])[0]
            if average_col > maxVal:  # If this mask is more whiter than any previous one, change prediction to this color
                prediction = color_name
                maxVal = average_col
        cv.imshow("RED MASK", masked_images["RED"])
        return prediction

    def getROI(self, img, contour_corner):
        x1 = int(contour_corner[0][0])
        x2 = int(contour_corner[1][0])
        x3 = int(contour_corner[2][0])
        x4 = int(contour_corner[3][0])
        y1 = int(contour_corner[0][1])
        y2 = int(contour_corner[1][1])
        y3 = int(contour_corner[2][1])
        y4 = int(contour_corner[3][1])
        top_left_x = min([x1, x2, x3, x4])
        top_left_y = min([y1, y2, y3, y4])
        bot_right_x = max([x1, x2, x3, x4])
        bot_right_y = max([y1, y2, y3, y4])
        return img[top_left_y:bot_right_y, top_left_x:bot_right_x]

    def getSquareROIs(contour_corners, img):
        cube_ROIs = []
        for corners in contour_corners:
            roi = getROI(img, corners)
            print('LENGTH ROI', len(roi))
            cube_ROIs.append(roi)
        return cube_ROIs

    def drawApproxRect(self, contour, img):  # Draws the approx Rect on given image and returns its corners
        rect = cv.minAreaRect(contour)  # Returns the origin, (width, height), angle of rotation of the minArea Rect
        box = cv.boxPoints(rect)  # Processes the rect information to get corners of the rectangle
        box = np.int0(box)
        cv.drawContours(img, [box], 0, (0, 255, 0), 2)
        print("BOX", box)
        return box

    def cropMinAreaRect(self, img, rect):
        # rotate img
        angle = rect[2]
        rows, cols = img.shape[0], img.shape[1]
        M = cv.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        img_rot = cv.warpAffine(img, M, (cols, rows))

        # rotate bounding box
        rect0 = (rect[0], rect[1], 0.0)
        box = cv.boxPoints(rect0)
        pts = np.int0(cv.transform(np.array([box]), M))[0]
        pts[pts < 0] = 0

        # crop
        img_crop = img_rot[pts[1][1]:pts[0][1], pts[1][0]:pts[2][0]]
        return img_crop
