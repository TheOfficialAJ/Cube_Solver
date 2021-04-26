import cuberesolver
import colordetector
import cv2 as cv
import Cube
import mainFrame

capture = cv.VideoCapture(0)
cubeResolver = cuberesolver.CubeResolver(capture, SCALE=1, mode=1)
colorDetector = colordetector.ColorDetector()
cube = Cube.Cube()
gui = mainFrame.App()

CubeImg = None

while True:
    gui.showImage(cubeResolver.getContourImage(), scale=0.6)
    gui.showAllFaces(cube.getAllFaces())
    gui.update()

    cubeResolver.generateSquareContours()
    keyPressed = cv.waitKey(50) & 0xFF
    if keyPressed == 27:
        break
    elif keyPressed == 13 and cubeResolver.isDetectionDone():  # If user pressed Enter and detection is done
        colors = []
        for cnt in cubeResolver.final_contours:
            _, color = colorDetector.getSquareColor(cubeResolver.image, cnt)
            colors.append(color)
        face_col = cube.addFace(colors)
        face_image = cube.drawFace(face_col)
        # cubeImg = cube.drawAllFaces()
        # cv.imshow("All faces", cubeImg)
        gui.showImage(face_image, frame=gui.face_panels[face_col], scale=0.5)

capture.release()
cv.destroyAllWindows()
