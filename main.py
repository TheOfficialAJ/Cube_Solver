import cv2 as cv

import Cube
import colordetector
import cuberesolver
import mainFrame

capture = cv.VideoCapture(0)
cubeResolver = cuberesolver.CubeResolver(capture, SCALE=1, mode=1)
colorDetector = colordetector.ColorDetector()
cube = Cube.Cube()
gui = mainFrame.App(colorDetector, cubeResolver, cube)


while True:
    # gui.showImage(cubeResolver.getContourImage(), scale=0.7)
    # gui.showAllFaces(cube.getAllFaces(sideLen=35))
    # gui.update()
    gui.run()

    cubeResolver.generateSquareContours()
    keyPressed = cv.waitKey(50) & 0xFF
    print("Key"+str(keyPressed))

    if keyPressed == 13 and cubeResolver.isDetectionDone():  # If user pressed Enter and detection is done
        colors = []
        for cnt in cubeResolver.final_contours:
            _, color = colorDetector.getSquareColor(cubeResolver.image, cnt)
            colors.append(color)
        face_col = cube.addFace(colors)
        face_image = cube.drawFace(face_col)
        # cubeImg = cube.drawAllFaces()
        # cv.imshow("All faces", cubeImg)
        gui.showImage(face_image, frame=gui.face_panels[face_col], scale=0.5)
    elif keyPressed == 27:
        break

capture.release()
cv.destroyAllWindows()
