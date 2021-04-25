import cv2 as cv
import numpy as np


class Cube:
    def __init__(self):
        self.cube_faces = {}
        self.COLORS = {'BLUE': (255, 0, 0), 'RED': (0, 0, 255), 'GREEN': (0, 255, 0), 'YELLOW': (0, 255, 255),
                       'ORANGE': (0, 128, 255), 'WHITE': (255, 255,
                                                          255)}

    def addFace(self, sorted_colors):
        mid_piece = sorted_colors[4]
        row = []
        face = []
        for index, col in enumerate(sorted_colors, 1):
            row.append(col)
            if index % 3 == 0:
                face.append(row)
                row = []
        self.cube_faces[mid_piece] = face

    def displayFace(self, face_color=None):
        if face_color is None:  # If no face name is supplied display any random face
            face_color = list(self.cube_faces.keys())[0]
            print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in self.cube_faces[face_color]]))
            self.drawFace(face_color)
        elif self.cube_faces[face_color]:
            print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in self.cube_faces[face_color]]))
            self.drawFace(face_color)

    def drawFace(self, face_color):
        blank = np.zeros((400, 400, 3), dtype='uint8')
        face = self.cube_faces[face_color]
        print("Face", face)
        sideLen = 50
        offsetX = 100
        offsetY = 100
        border = 5  # Distance (in pixels) between each square
        for i in range(3):
            startY = int(sideLen * i) + offsetY
            endY = startY + sideLen - border
            for j in range(3):
                startX = int(sideLen * j) + offsetX
                endX = startX + sideLen - border
                square_color = self.COLORS[face[i][j]]
                blank = cv.rectangle(blank, (startX, startY), (endX, endY), square_color, -1)

        cv.imshow(face_color + " Face", blank)

    def displayAllFaces(self):
        for face_name, face in self.cube_faces.items():
            self.displayFace(face_name)
            print("===================================================")
