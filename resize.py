import cv2 as cv


def rescale(img, scale=0.15):
    # Resizing Image to fit it on screen
    height = int(img.shape[0] * scale)
    width = int(img.shape[1] * scale)
    img = cv.resize(img, (width, height), interpolation=cv.INTER_AREA)
    return img


def resize(ing, x, y):
    img = cv.resize(ing, (x, y), interpolation=cv.INTER_AREA)
    return img
