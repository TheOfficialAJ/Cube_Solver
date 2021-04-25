import cuberesolver
import colordetector
import cv2 as cv
import Cube

capture = cv.VideoCapture(0)
cubeResolver = cuberesolver.CubeResolver(capture, SCALE=1, mode=1)
colorDetector = colordetector.ColorDetector()
cube = Cube.Cube()

while True:
    cubeResolver.generateSquareContours()
    keyPressed = cv.waitKey(50) & 0xFF
    if keyPressed == 27:
        break
    elif keyPressed == 13 and cubeResolver.isDetectionDone():  # If user pressed Enter and detection is done
        colors = []
        for cnt in cubeResolver.final_contours:
            _, color = colorDetector.getSquareColor(cubeResolver.image, cnt)
            colors.append(color)
        cube.addFace(colors)
        cube.displayFace()
cube.displayAllFaces()

capture.release()
cv.destroyAllWindows()
