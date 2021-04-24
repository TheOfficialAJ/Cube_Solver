import cuberesolver
import colordetector
import cv2 as cv

capture = cv.VideoCapture(0)
cubeResolver = cuberesolver.CubeResolver(capture, SCALE=1, mode=1)
colorDetector = colordetector.ColorDetector()
while True:
    cubeResolver.generateSquareContours()
    input = cv.waitKey(50) & 0xFF
    if input == 27:
        break
    elif input == 13 and cubeResolver.isDetectionDone():  # If user pressed Enter and detection is done
        print("COMPLETED")
        print(colorDetector.predictColor(cubeResolver.image, cubeResolver.final_contours[0]))
        for cnt in cubeResolver.final_contours:
            img = colorDetector.showSquareColors(cubeResolver.image, cnt)

capture.release()
cv.destroyAllWindows()
