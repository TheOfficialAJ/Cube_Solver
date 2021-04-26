import cv2 as cv
import numpy as np
import colour as clr
from resize import resize
import imutils.contours
import contoursorter


def getAverageColor(img_ROI):  # Returns the average color of ROI in LAB color space
    l = []
    a = []
    b = []
    img_ROI = cv.cvtColor(img_ROI, cv.COLOR_BGR2Lab)
    for x in range(img_ROI.shape[1]):
        for y in range(img_ROI.shape[0]):
            if l != 0:
                l.append((img_ROI[y][x])[0])
                a.append((img_ROI[y][x])[1])
                b.append((img_ROI[y][x])[2])

    return round(np.mean(l), 2), round(np.mean(a), 2), round(np.mean(b), 2)


class CubeResolver:
    def __init__(self, image, SCALE, mode):
        self.approx_contours = []
        self.paused = False
        self.image = image
        if mode == 1:
            self.capture = image
            _, self.image = self.image.read()
        else:
            self.capture = None
        self.mode = mode
        # self.final_contour_centres = []
        self.final_contours_corners = []
        self.final_contours = []
        self.square_contours = []
        self.SCALE = SCALE
        self.tempImg = self.image.copy()
        cv.namedWindow("Final Contours", cv.WINDOW_NORMAL)
        cv.setMouseCallback("Final Contours", self.onClick)
        self.image = resize(self.image, self.SCALE)

    def onClick(self, event, x, y, flags, param):
        if self.isDetectionDone() and event == cv.EVENT_LBUTTONDBLCLK:
            print("Detection Complete")
            print("Detection Complete")

        elif event == cv.EVENT_LBUTTONUP:
            print("Click Registered")
            if not self.paused:
                self.paused = True
            else:
                self.paused = False

    def resetContours(self):
        self.approx_contours = []
        # self.final_contour_centres = []
        self.final_contours_corners = []
        self.final_contours = []
        self.square_contours = []
        self.tempImg = self.image.copy()

    def sort_contours(self, img):
        conts = self.final_contours
        if len(self.final_contours) != 9:
            print("ERROR Detection not proper")
            return

        contours = contoursorter.sort_contours(conts, method="top-to-bottom")

        row = []
        cube_rows = []
        for (i, cnt) in enumerate(contours, 1):
            row.append(cnt)
            if i % 3 == 0:
                contours = contoursorter.sort_contours(row, method="left-to-right")
                cube_rows.append(contours)
                row = []
        del row
        num = 0

        self.final_contours = []  # Sorts the final_contours

        for row in cube_rows:
            for cnt in row:
                self.final_contours.append(cnt)

        # for row in cube_rows:
        #     for cnt in row:
        #         x, y, w, h = cv.boundingRect(cnt)
        #         cv.putText(img, "#" + str(num), (x + 5, y + 10), cv.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
        #         num += 1

        for i in range(len(self.final_contours)):
            cnt = self.final_contours[i]
            cx, cy = self.getContourCentre(cnt)
            text = '#' + str(i)
            cv.putText(img, text, (cx - 3, cy + 2), cv.FONT_HERSHEY_DUPLEX, 0.3, (0, 0, 0), 1)

    def distance(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return dist

    def getMinDistance(self, cnt, contour_list):
        distance = []
        x1, y1 = self.getContourCentre(cnt)
        for c in contour_list:
            x2, y2 = self.getContourCentre(c)
            dist = self.distance((x1, y1), (x2, y2))
            if dist != 0:
                distance.append(dist)
        return min(distance)

    def isDetectionDone(self):
        if len(self.final_contours) != 9:
            return False

        dist_thresh = 2
        min_distances = []
        for cnt in self.final_contours:
            min_dist = round(self.getMinDistance(cnt, self.final_contours), 2)
            min_distances.append(min_dist)
        mean_dist = sum(min_distances) / len(min_distances)
        for dist in min_distances:
            if dist + dist_thresh > dist > mean_dist - dist_thresh:
                return True
            else:
                return False

    def prepareImg(self):
        if self.capture is not None:
            _, self.image = self.capture.read()
        # Grayscale
        gray = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
        # cv.imshow("Gray", gray)

        # Gaussian Blur (GRAY) (For calculating canny threshold)
        blur_gray = cv.GaussianBlur(gray, (3, 3), 0)

        # Gaussian Blur
        blur = cv.GaussianBlur(self.image, (3, 3), 0)
        # cv.imshow("Blurred", blur)

        # Canny
        high_thresh, thresh_im = cv.threshold(blur_gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        print("Thresholds", high_thresh)
        low_thresh = high_thresh // 2
        canny = cv.Canny(blur, low_thresh, high_thresh)
        # cv.imshow("Canny", canny)

        # Dilation
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv.dilate(canny, kernel, iterations=2)
        # cv.imshow("Dilated", dilated)
        return dilated

    def getContourCentre(self, cnt):
        M = cv.moments(cnt)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        return cx, cy

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

    def getAverageColor(self, img_roi):  # Returns the average color of ROI in LAB color space
        l = []
        a = []
        b = []
        img_ROI = cv.cvtColor(img_roi, cv.COLOR_BGR2Lab)
        for x in range(img_ROI.shape[1]):
            for y in range(img_ROI.shape[0]):
                l.append((img_ROI[y][x])[0])
                a.append((img_ROI[y][x])[1])
                b.append((img_ROI[y][x])[2])

        return round(np.mean(l)), round(np.mean(a)), round(np.mean(b))

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

    def getSquareROIs(self, contour_corners, img):
        cube_ROIs = []
        for corners in contour_corners:
            roi = self.getROI(img, corners)
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

    def getContourImage(self):
        img = self.image.copy()
        cv.drawContours(img, self.final_contours, -1, (0,255,0), 2)
        return img

    def generateSquareContours(self):
        if self.paused:
            return

        self.resetContours()
        dilated = self.prepareImg()
        contours, hierarchy = cv.findContours(dilated, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        allContImg = self.image.copy()
        cv.drawContours(allContImg, contours, -1, (0, 0, 255), 2)
        for i in range(len(contours)):
            cnt = contours[i]
            epsilon = 0.1 * cv.arcLength(cnt, True)
            # rect = cv.minAreaRect(cnt)
            # box = cv.boxPoints(rect)
            # box = np.int0(box)
            # cv.drawContours(contImg, [box], 0, (255, 0, 0), 2)
            # cv.imshow("Contours All", contImg)
            approx = cv.approxPolyDP(cnt, epsilon, True)
            self.approx_contours.append(approx)

            # cv.rectangle(self.image, (x, y), (x + w, y + h), (255, 0, 0), 1)

        # cv.drawContours(self.tempImg, self.approx_contours, -1, (0, 0, 255), 2)  # Draws the contours
        # cv.imshow("Approx Contours", self.tempImg)
        contImg = self.image.copy()
        for i in range(len(self.approx_contours)):
            cnt = self.approx_contours[i]
            area = cv.contourArea(cnt)

            side = area ** (1 / 2)
            perimeter = cv.arcLength(cnt, True)
            # threshold = 375 * self.SCALE ** 2
            threshold = 20 * (self.SCALE ** 2)
            print("AREA", area)
            if (area > 600 * self.SCALE ** 2) and 4 * side - threshold < perimeter < 4 * side + threshold:
                self.square_contours.append(cnt)
                x, y, w, h = cv.boundingRect(cnt)
                print(x, y, h, w)
                print(cv.arcLength(cnt, True))
                rect = cv.minAreaRect(cnt)
                box = cv.boxPoints(rect)
                box = np.int0(box)
                cv.drawContours(contImg, [box], 0, (0, 255, 0), 2)
                # cv.imshow("Contours Square", contImg)

        areas = [cv.contourArea(cnt) for cnt in self.square_contours]
        print(areas)
        # To remove all contours which are outliers in area. Uses IQR

        if areas:  # Only do this if the areas list has some elements to avoid Exceptions
            Q1 = np.quantile(areas, 0.25)  # Some maths stuff named "Inter Quartile Range" to remove Outliers
            Q3 = np.quantile(areas, 0.75)
            IQR = Q3 - Q1
            for i in range(len(self.square_contours)):
                cnt = self.square_contours[i]
                cnt_area = cv.contourArea(cnt)
                print("area", cnt_area)
                if cnt_area < Q1 - (1.5 * IQR) or cnt_area > Q3 + (1.5 * IQR):
                    print("WORNG CONTOUR")
                    continue
                else:
                    self.final_contours_corners.append(self.drawApproxRect(cnt, contImg))
                    self.final_contours.append(cnt)

        # for cnt in self.final_contours:
        #     self.final_contour_centres.append(self.getContourCentre(cnt))
        #     # cv.circle(self.image, (cx, cy), 4, (0, 0, 255), -1)

        # Removes nested contours
        i = 0
        while i < len(self.final_contours):
            j = 0
            while j < len(self.final_contours):
                if i >= len(self.final_contours):
                    break
                corners1 = self.getDiagonalCorners(self.final_contours_corners[i])
                corners2 = self.getDiagonalCorners(self.final_contours_corners[j])
                # side_len1 = cv.arcLength(final_contours[i], True) // 4
                # side_len2 = cv.arcLength(final_contours[j], True) // 4
                if i != j:
                    if (corners1[0][0] < corners2[0][0] and corners1[0][1] < corners2[0][1]) and (
                            corners1[1][0] > corners2[1][0] and corners1[1][1] > corners2[1][1]):
                        print("CONTOUR REMOVED", j)
                        del self.final_contours_corners[j]
                        del self.final_contours[j]
                        # del self.final_contour_centres[j]
                    # elif distance(square_centers[i], square_centers[j]) < (side_len1 / 2 + side_len2 / 2):
                    #     print("CENTER DISTANCE TOO LOW")
                    #     del final_contours_corners[j]
                    #     del final_contours[j]
                    #     del square_centers[j]
                j += 1
            i += 1
        finalContImg = self.image.copy()

        self.sort_contours(finalContImg)
        if self.isDetectionDone():
            self.paused = True

        cv.drawContours(finalContImg, self.final_contours, -1, (0, 255, 0), 2)
        cv.imshow("Final Contours", finalContImg)

        # mask = np.zeros(self.image.shape[:2], dtype='uint8')  # Note: Don't use mask = np.zeros_like(self.image) here as it gives error
        # cv.drawContours(mask, self.final_contours, -1, (255,255,255), -1)
        # cv.imshow("MASK", mask)
        # print(mask.shape)
        # print(self.image.shape)
        # masked = cv.bitwise_and(self.image, self.image, mask=mask)
        # print("Color is", getAverageColor(masked))
        # cv.imshow("CONTOUR AS MASk", masked)
