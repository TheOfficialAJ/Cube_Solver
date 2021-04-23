import cv2 as cv
import numpy as np
import functools

# def sort_contours(cnts, method="left-to-right"):
#     # initialize the reverse flag and sort index
#     reverse = False
#     i = 0
#     # handle if we need to sort in reverse
#     if method == "right-to-left" or method == "bottom-to-top":
#         reverse = True
#     # handle if we are sorting against the y-coordinate rather than
#     # the x-coordinate of the bounding box
#     if method == "top-to-bottom" or method == "bottom-to-top":
#         i = 1
#     # construct the list of bounding boxes and sort them from top to
#     # bottom
#     boundingBoxes = [cv2.boundingRect(c) for c in cnts]
#     (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
#                                         key=lambda b: b[1][i], reverse=reverse))
#     # return the list of sorted contours and bounding boxes
#     return cnts, boundingBoxes

def greaterX(a, b):
    momA = cv.moments(a)
    (xa,ya) = int(momA['m10']/momA['m00']), int(momA['m01']/momA['m00'])

    momB = cv.moments(b)
    (xb,yb) = int(momB['m10']/momB['m00']), int(momB['m01']/momB['m00'])
    if xa > xb:
        return 1

    if xa == xb:
        return 0
    else:
        return -1

def greaterY(a, b):
    momA = cv.moments(a)
    (xa,ya) = int(momA['m10']/momA['m00']), int(momA['m01']/momA['m00'])

    momB = cv.moments(b)
    (xb,yb) = int(momB['m10']/momB['m00']), int(momB['m01']/momB['m00'])
    if ya > yb:
        return 1

    if ya == yb:
        return 0
    else:
        return -1


def sort_contours(cnts, method="left-to-right"):
    if method == "left-to-right" or method == "right-to-left":
        cnts.sort(key=functools.cmp_to_key(greaterX))
        return cnts
    if method == "top-to-bottom" or method == "bottom-to-top":
        cnts.sort(key=functools.cmp_to_key(greaterY))
        return cnts