import cv2 as cv
import numpy as np

import resize

xi = yi = xf = yf = 0
img_roi = None


def meanColor(image):
    color_b = []
    color_g = []
    color_r = []
    for x in range(image.shape[1]):
        for y in range(image.shape[0]):
            color_b.append((image[y][x])[0])
            color_g.append((image[y][x])[1])
            color_r.append((image[y][x])[2])

    return (np.mean(color_b), np.mean(color_g), np.mean(color_r))


def displayPixelVal(event, x, y, flags, param):
    global xi, yi, xf, yf
    global mod_img, img_roi
    if event == cv.EVENT_LBUTTONDOWN:
        print("displayPixelVal called")
        print(mod_img[y, x])
        img_lab = cv.cvtColor(img, cv.COLOR_BGR2Lab)
        print("COLOR:", img_lab[y][x])
        xi = x
        yi = y
        cv.putText(mod_img, str(mod_img[y][x]), (x, y), cv.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255), 2)
    elif event == cv.EVENT_LBUTTONUP:
        xf = x
        yf = y
        if xi != xf and yi != yf and xi < xf and yi < yf:
            print(xi, yi, xf, yf)
            cv.destroyWindow('ROI')
            img_roi = None
            img_roi = img[yi:yf, xi:xf]
            print(img_roi.shape[:2])
            # cv.imshow("Image ROI", img_roi)
            cv.rectangle(mod_img, (xi, yi), (xf, yf), (0, 255, 0), 1)
            color_avg = meanColor(img_roi)
            color_avg_str = f"({round(color_avg[0], 2)}, {round(color_avg[1], 2)}, {round(color_avg[2], 2)})"
            print("Mean Color:", meanColor(img_roi))
            cv.putText(mod_img, color_avg_str, ((xi + xf) // 2, (yi + yf) // 2), cv.FONT_HERSHEY_TRIPLEX, 0.5,
                       (0, 255, 0), 1)
        # cv.imshow("image", img)


vid = cv.VideoCapture(0)
while True:
    ret, img = vid.read()
    if cv.waitKey(1) == ord('q'):
        break
    cv.imshow("RGB IMAGE", img)
    img = cv.cvtColor(img, cv.COLOR_BGR2Lab)
    l, a, b = cv.split(img)
    cv.setMouseCallback('Image', displayPixelVal)
    cv.setMouseCallback('RGB IMAGE', displayPixelVal)
    mod_img = img.copy()  # This is the image that will be displayed wih the rectangles to protect the original one
    while True:
        if img_roi is not None:
            cv.imshow("ROI", img_roi)
        cv.imshow("Image", mod_img)
        k = cv.waitKey(20) & 0xFF
        if k == 27:
            break
        elif k == 114:
            mod_img = img.copy()

    # cv.imshow("L", l)
    # cv.imshow("A", a)
    # cv.imshow("B", b)
    # print(f"New size {img.shape[1]}x{img.shape[0]}")

    # cv.imshow("Image", mod_img)

# img = cv.imread("../OpenCV Image Processing/cube1.jpg")


cv.destroyAllWindows()
