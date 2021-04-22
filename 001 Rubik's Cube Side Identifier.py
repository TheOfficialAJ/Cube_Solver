import cv2 as cv
import numpy as np
import colour as clr
from resize import resize

SCALE = 0.15
COLORS = {'BLUE': (240, 250, 3), 'RED': (0, 0, 255), 'GREEN': (0, 255, 0), 'YELLOW': (0, 255, 255),
          'ORANGE': (0, 165, 255)}

COLOR_RANGES = {'YELLOW': [(20, 100, 175), (255, 135, 215)], 'ORANGE': [(100, 145, 140), (230, 180, 190)],
                'RED': [(50, 150, 120), (120, 190, 170)], 'GREEN': [(40, 55, 140), (220, 110, 180)],
                'BLUE': [(40, 145, 50), (220, 180, 110)]}

cubeImg = cv.imread("Cube Images/cube5.jpg")
cubeImg = resize(cubeImg, SCALE)
contImg = cubeImg.copy()
notModImg = cubeImg.copy()
# cubeImg = cv.cvtColor(cubeImg, cv.COLOR_BGR2HLS)
cv.imshow("Image", cubeImg)
# Creating blank image
blank_img = np.zeros((cubeImg.shape[0], cubeImg.shape[1], 3), dtype="uint8")


def getDiagonalCorners(contour_corners):
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


def distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return dist


def getAverageColor(img_ROI):  # Returns the average color of ROI in LAB color space
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


def predictColor(image, corners, color_ranges):
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


def getROI(img, contour_corner):
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


def drawApproxRect(contour, img):  # Draws the approx Rect on given image and returns its corners
    rect = cv.minAreaRect(contour)  # Returns the origin, (width, height), angle of rotation of the minArea Rect
    box = cv.boxPoints(rect)  # Processes the rect information to get corners of the rectangle
    box = np.int0(box)
    cv.drawContours(img, [box], 0, (0, 255, 0), 2)
    print("BOX", box)
    return box


def cropMinAreaRect(img, rect):
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


# **********************************************************************************************************************
# **********************************************************************************************************************

# Grayscale
gray = cv.cvtColor(cubeImg, cv.COLOR_BGR2GRAY)
cv.imshow("Gray", gray)

# Gaussian Blur (GRAY)
blur_gray = cv.GaussianBlur(gray, (3, 3), 0)

# Gaussian Blur
blur = cv.GaussianBlur(cubeImg, (3, 3), 0)
cv.imshow("Blurred", blur)

# Canny
sigma = 0.33
high_thresh, thresh_im = cv.threshold(blur_gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
print("Thresholds", high_thresh)
low_thresh = high_thresh // 2
canny = cv.Canny(blur, low_thresh, high_thresh)
cv.imshow("Canny", canny)

# Dilation
kernel = np.ones((3, 3), np.uint8)
dilated = cv.dilate(canny, kernel, iterations=2)
cv.imshow("Dilated", dilated)

contours, hierarchy = cv.findContours(dilated, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

approx_contours = []
allContImg = cubeImg.copy()
cv.drawContours(allContImg, contours, -1, (0, 0, 255), 2)
for i in range(len(contours)):
    cnt = contours[i]
    epsilon = 0.1 * cv.arcLength(cnt, True)
    rect = cv.minAreaRect(cnt)
    # box = cv.boxPoints(rect)
    # box = np.int0(box)
    # cv.drawContours(contImg, [box], 0, (255, 0, 0), 2)
    # cv.imshow("Contours All", contImg)
    approx = cv.approxPolyDP(cnt, epsilon, True)
    approx_contours.append(approx)

    # cv.rectangle(cubeImg, (x, y), (x + w, y + h), (255, 0, 0), 1)
cv.drawContours(cubeImg, approx_contours, -1, (0, 0, 255), 2)
square_contours = []
for i in range(len(approx_contours)):
    cnt = approx_contours[i]
    area = cv.contourArea(cnt)

    side = area ** (1 / 2)
    perimeter = cv.arcLength(cnt, True)
    # Q1 = np.quantile(approx_contours, 0.25)
    # Q3 = np.quantile(approx_contours, 0.75)
    threshold = 375 * SCALE ** 2
    if (area > 90000 * SCALE ** 2) and 4 * side - threshold < perimeter < 4 * side + threshold:
        square_contours.append(cnt)
        x, y, w, h = cv.boundingRect(cnt)
        print(x, y, h, w)
        print(cv.arcLength(cnt, True))
        rect = cv.minAreaRect(cnt)
        box = cv.boxPoints(rect)
        box = np.int0(box)
        cv.drawContours(contImg, [box], 0, (0, 255, 0), 2)
        cv.imshow("Contours Square", contImg)

areas = [cv.contourArea(cnt) for cnt in square_contours]
print(areas)
# To remove all contours which are outliers in area. Uses IQR

Q1 = np.quantile(areas, 0.25)  # Some maths stuff named "Inter Quartile Range" to remove Outliers
Q3 = np.quantile(areas, 0.75)
IQR = Q3 - Q1
final_contours = []
final_contours_corners = []
for i in range(len(square_contours)):
    cnt = square_contours[i]
    cnt_area = cv.contourArea(cnt)
    print("area", cnt_area)
    if cnt_area < Q1 - (1.5 * IQR) or cnt_area > Q3 + (1.5 * IQR):
        print("WORNG CONTOUR")
        continue
    else:
        final_contours_corners.append(drawApproxRect(cnt, contImg))
        final_contours.append(cnt)

square_centers = []
for cnt in final_contours:
    M = cv.moments(cnt)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    print("Color is", cubeImg[cy][cx])
    # cv.circle(cubeImg, (cx, cy), 4, (0, 0, 255), -1)
    square_centers.append((cx, cy))

    i = 0
    j = 0

while i < len(final_contours):
    resX = cubeImg.shape[1]
    resY = cubeImg.shape[0]
    j = 0
    while j < len(final_contours):
        corners1 = getDiagonalCorners(final_contours_corners[i])
        corners2 = getDiagonalCorners(final_contours_corners[j])
        # side_len1 = cv.arcLength(final_contours[i], True) // 4
        # side_len2 = cv.arcLength(final_contours[j], True) // 4
        if i != j:
            if (corners1[0][0] < corners2[0][0] and corners1[0][1] < corners2[0][1]) and (
                    corners1[1][0] > corners2[1][0] and corners1[1][1] > corners2[1][1]):
                print("CONTOUR REMOVED", j)
                del final_contours_corners[i]
                del final_contours[i]
                del square_centers[i]
            # elif distance(square_centers[i], square_centers[j]) < (side_len1 / 2 + side_len2 / 2):
            #     print("CENTER DISTANCE TOO LOW")
            #     del final_contours_corners[j]
            #     del final_contours[j]
            #     del square_centers[j]
        j += 1
    i += 1

square_ROIs = getSquareROIs(final_contours_corners, notModImg)
for i in range(len(square_ROIs)):
    avg_color = getAverageColor(square_ROIs[i])
    # print("AVG COLOR", avg_color)
    cv.putText(cubeImg,
               "ROI" + str(i), (square_centers[i][0] - 10, square_centers[i][1]),
               cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 15), 1)
    cv.putText(cubeImg,
               str(avg_color), (square_centers[i][0] - 25, square_centers[i][1] + 15),
               cv.FONT_HERSHEY_DUPLEX, 0.35, (255, 0, 0), 1)
    # print("Final Contours:", final_contours)
    # test_area = cubeImg[square_centers[7][1]:square_centers[0][1] + 10, square_centers[7][0]:square_centers[7][0] + 10]
    # test_area = resize(test_area, 5)
    # cv.imshow("Test Area", test_area)
# print(test_area)SQ
# getClosestColor(test_area)


cv.drawContours(blank_img, final_contours, -1, (0, 255, 255), 1)
cv.imshow("Final Contours", blank_img)
cv.imshow("SQUARE ROI", cubeImg)
# predict_index = 7
# print("Color is", predictColor(notModImg, final_contours_corners[predict_index], COLOR_RANGES))
predictionImage = notModImg.copy()

for i in range(len(final_contours_corners)):
    cnt_corners = final_contours_corners[i]
    color = predictColor(predictionImage, final_contours_corners[i], COLOR_RANGES)
    prediction_corners = getDiagonalCorners(cnt_corners)
    if color is not None:
        # cv.rectangle(predictionImage, prediction_corners[0], prediction_corners[1], COLORS[color], 2)
        cv.drawContours(predictionImage, final_contours, i, COLORS[color], 2)
cv.imshow("PREDICTION ROI", predictionImage)
# prediction_corners = getDiagonalCorners(final_contours_corners[predict_index])

# cv.rectangle(predictionImage, prediction_corners[0], prediction_corners[1], (0, 255, 0), 2)


cv.waitKey(0)
