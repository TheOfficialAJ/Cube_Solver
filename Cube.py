import time

import cv2 as cv
import numpy as np
import rubik_solver.utils

'''              3
*            =========
*            ====Y====
*            =========
*      1         2         5         6
*  ========= ========= ========= =========
*  ====B==== ====R==== ====G==== ====O====
*  ========= ========= ========= =========
*                4
*            =========
*            ====W====
*            =========
'''


class Cube:
    def __init__(self):
        self.COLORS = {'BLUE': (255, 0, 0), 'RED': (0, 0, 255), 'GREEN': (0, 255, 0), 'YELLOW': (0, 255, 255),
                       'ORANGE': (0, 128, 255), 'WHITE': (255, 255,
                                                          255), 'GRAY': (128, 128, 128)}
        self.color_pos = {'BLUE': (0, 1), 'RED': (1, 1), 'GREEN': (2, 1), 'YELLOW': (1, 0), 'ORANGE': (3, 1), 'WHITE': (1, 2)}

        self.cube_faces = {}
        for color in self.COLORS.keys():  # Initializes all faces to have just the center square colored
            if color != 'GRAY':
                self.cube_faces[color] = [['GRAY', 'GRAY', 'GRAY'] for i in range(3)]
                self.cube_faces[color][1][1] = color
            print("CUBE FACES", self.cube_faces)

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
        return mid_piece

    # TODO Add the rotation functionality
    def rotateFace(self, face_name):
        face_arr = self.cube_faces[face_name]
        self.cube_faces[face_name] = list(zip(*face_arr[::-1]))
        print("Face Rotated")

    def displayFace(self, face_color=None):
        if face_color is None:  # If no face name is supplied display any random face
            notNoneFaces = {k: v for k, v in self.cube_faces.items() if v is not None}
            face_color = list(notNoneFaces.keys())[0]
            print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in self.cube_faces[face_color]]))
            self.drawFace(face_color)
        elif self.cube_faces[face_color]:
            print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in self.cube_faces[face_color]]))
            self.drawFace(face_color)

    def drawFace(self, face_color, offsetX=10, offsetY=10, canvas_img=None, square_len=50, border=2):
        if face_color == 'GRAY':
            print("No gray face")
            return

        if canvas_img is None:
            width = square_len * 3 + offsetX * 2
            height = square_len * 3 + offsetY * 2
            print("width", width)
            canvas_img = np.zeros((width, height, 3), dtype='uint8')

        face = self.cube_faces[face_color]

        for i in range(3):
            startY = int(square_len * i) + offsetY
            endY = startY + square_len
            for j in range(3):
                startX = int(square_len * j) + offsetX
                endX = startX + square_len
                square_color = self.COLORS[face[i][j]]
                cv.rectangle(canvas_img, (startX, startY), (endX, endY), square_color, -1)  # Draws the square
                cv.rectangle(canvas_img, (startX, startY), (endX, endY), (0, 0, 0), border)  # Draws the border in black
        return canvas_img

    def displayAllFaces(self):
        for face_name, face in self.cube_faces.items():
            self.displayFace(face_name)
            print("===================================================")

    # def drawAllFaces(self):
    #     blank = np.zeros((1000, 1000, 3), dtype='uint8')
    #     offset = 180
    #     for color in self.COLORS:
    #         if color != 'GRAY':
    #             offsetX = int(self.color_pos[color][0] * offset)
    #             offsetY = int(self.color_pos[color][1] * offset)
    #             blank = self.drawFace(color, offsetX=offsetX, offsetY=offsetY, canvas_img=blank)
    #     return blank

    def getAllFaces(self, sideLen=45):
        face_img_dict = {}
        for color in self.COLORS:
            if color != 'GRAY':
                face = self.cube_faces[color]
                img = self.drawFace(color, square_len=sideLen, offsetX=5, offsetY=5)
                face_img_dict[color] = img

        return face_img_dict

    def getSolution(self):
        return self._getCubeSolution()

    def isSolveable(self):
        color_nums = {col: 0 for col in self.COLORS.keys()}
        del color_nums['GRAY']
        for face in self.cube_faces.values():
            print("face:", face)
            for row in face:
                for col in row:
                    print("row", row)
                    if col == 'GRAY':
                        return False
                    else:
                        color_nums[col] += 1
        print("Color_nums:", color_nums)
        print("color_cums values:", color_nums.values())
        if len(set(color_nums.values())) == 1:
            return True
        else:
            return False

    def _getCubeSolution(self):
        cubeStr = ""
        print("gray in cube", 'GRAY' in self.cube_faces)
        for i in range(3):
            for j in range(3):
                cubeStr += self.cube_faces['YELLOW'][i][j][0]

        # for i in range(3):
        #     for face in ['BLUE', 'RED', "GREEN", 'ORANGE']:
        #         cubeStr += "".join([col[0] for col in self.cube_faces[face][i]])

        for face in ['BLUE', 'RED', "GREEN", 'ORANGE']:
            for i in range(3):
                cubeStr += "".join([col[0] for col in self.cube_faces[face][i]])

        for i in range(3):
            for j in range(3):
                cubeStr += self.cube_faces['WHITE'][i][j][0]
        cubeStr = cubeStr.lower()

        try:
            solution = rubik_solver.utils.solve(cubeStr, method='Kociemba')
            print(solution)
            return solution
        except:
            print("Error Invalid Cube")
