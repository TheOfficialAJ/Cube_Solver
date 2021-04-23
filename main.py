import cuberesolver
import cv2 as cv

capture = cv.VideoCapture(0)
cubeResolver = cuberesolver.CubeResolver(capture, 1, 1)

while True:
    cubeResolver.generateSquareContours()
    if cv.waitKey(100) & 0xFF == 27:
        break

capture.release()
cv.destroyAllWindows()
