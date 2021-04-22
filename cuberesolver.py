import cv2 as cv
import numpy as np
import colour as clr
from resize import resize


class CubeResolver:
    def __init__(self, image, SCALE, mode):
        self.image = image
        if mode == 1:
            self.capture = image
            _, self.image = self.image.read()
        else:
            self.capture = None
        self.mode = mode
        self.square_centers = []
        self.final_contours_corners = []
        self.final_contours = []
        self.square_contours = []
        self.SCALE = SCALE
        self.tempImg = self.image.copy()
        self.image = resize(self.image, self.SCALE)

    def resetContours(self):
        self.square_centers = []
        self.final_contours_corners = []
        self.final_contours = []
        self.square_contours = []
        self.tempImg = self.image.copy()

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
        sigma = 0.33
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

    def predictColor(self, image, corners, color_ranges):
        diagCorners = self.getDiagonalCorners(corners)
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
            average_col = self.getAverageColor(masked_images[color_name])[0]
            if average_col > maxVal:  # If this mask is more whiter than any previous one, change prediction to this color
                prediction = color_name
                maxVal = average_col
        # cv.imshow("RED MASK", masked_images["RED"])
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

    def generateSquareContours(self):
        self.resetContours()
        dilated = self.prepareImg()
        contours, hierarchy = cv.findContours(dilated, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        self.approx_contours = []
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

        cv.drawContours(self.tempImg, self.approx_contours, -1, (0, 0, 255), 2)  # Draws the contours
        cv.imshow("Approx Contours", self.tempImg)
        contImg = self.image.copy()
        for i in range(len(self.approx_contours)):
            cnt = self.approx_contours[i]
            area = cv.contourArea(cnt)

            side = area ** (1 / 2)
            perimeter = cv.arcLength(cnt, True)
            threshold = 375 * self.SCALE ** 2
            print("AREA", area)
            if (area > 800 * self.SCALE ** 2) and 4 * side - threshold < perimeter < 4 * side + threshold:
                self.square_contours.append(cnt)
                x, y, w, h = cv.boundingRect(cnt)
                print(x, y, h, w)
                print(cv.arcLength(cnt, True))
                rect = cv.minAreaRect(cnt)
                box = cv.boxPoints(rect)
                box = np.int0(box)
                cv.drawContours(contImg, [box], 0, (0, 255, 0), 2)
                cv.imshow("Contours Square", contImg)

        areas = [cv.contourArea(cnt) for cnt in self.square_contours]
        print(areas)
        # To remove all contours which are outliers in area. Uses IQR

        if areas:
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

        for cnt in self.final_contours:
            M = cv.moments(cnt)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            print("Color is", self.image[cy][cx])
            # cv.circle(self.image, (cx, cy), 4, (0, 0, 255), -1)
            self.square_centers.append((cx, cy))

        i = 0
        while i < len(self.final_contours):
            resX = self.image.shape[1]
            resY = self.image.shape[0]
            j = 0
            while j < len(self.final_contours):
                corners1 = self.getDiagonalCorners(self.final_contours_corners[i])
                corners2 = self.getDiagonalCorners(self.final_contours_corners[j])
                # side_len1 = cv.arcLength(final_contours[i], True) // 4
                # side_len2 = cv.arcLength(final_contours[j], True) // 4
                if i != j:
                    if (corners1[0][0] < corners2[0][0] and corners1[0][1] < corners2[0][1]) and (
                            corners1[1][0] > corners2[1][0] and corners1[1][1] > corners2[1][1]):
                        print("CONTOUR REMOVED", j)
                        del self.final_contours_corners[i]
                        del self.final_contours[i]
                        del self.square_centers[i]
                    # elif distance(square_centers[i], square_centers[j]) < (side_len1 / 2 + side_len2 / 2):
                    #     print("CENTER DISTANCE TOO LOW")
                    #     del final_contours_corners[j]
                    #     del final_contours[j]
                    #     del square_centers[j]
                j += 1
            i += 1
            finalContImg = self.image.copy()
            cv.drawContours(finalContImg, self.final_contours, -1, (0, 255, 0), 2)
            cv.imshow("Final Contours", finalContImg)
