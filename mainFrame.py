import tkinter as tk

import cv2 as cv
import keyboard
from PIL import Image, ImageTk

import Cube
import colordetector
import cuberesolver
from resize import rescale, resize


def disable_btn(btn, disable_img):
    btn.config(image=disable_img, relief='sunken')


def enable_btn(btn, normal_img):
    btn.config(image=normal_img, relief='raised')


class App(tk.Tk):
    def __init__(self, color_detector: colordetector.ColorDetector, cube_resolver: cuberesolver.CubeResolver, cube: Cube.Cube):
        super().__init__()
        self.title("Cube Solver")
        self.window_bg = "#322f3d"
        self.geometry('1080x720')
        self.configure(bg=self.window_bg)

        # Initializing class variables
        self.colorDetector = color_detector
        self.cubeResolver = cube_resolver
        self.cube_obj = cube
        self.doAutoDetect = False

        # Button Graphics
        self.predict_img = tk.PhotoImage(file="GUI Assets/button_generate-solution.png")
        self.predict_img_disabled = tk.PhotoImage(file="GUI Assets/button_generate-solution_disabled.png")
        self.predict_img_active = tk.PhotoImage(file="GUI Assets/button_generate-solution_active.png")
        self.predict_img_hover = tk.PhotoImage(file="GUI Assets/button_generate-solution_hover.png")
        self.fp_container_img = tk.PhotoImage(file="GUI Assets/rounded.php.png")

        # ---------------------- Initializing Objects ----------------------

        # Camera panel and face panels
        self.camera_feed_panel = tk.Canvas(self, borderwidth=2, bg='black', width=640 * 0.77, height=480 * 0.77)

        # Face Panels
        self.face_panels = {'RED': None, 'BLUE': None, 'GREEN': None, 'YELLOW': None, 'WHITE': None, 'ORANGE': None}
        self.face_panel_container = tk.LabelFrame(self, borderwidth=1)
        self.face_panel_container.configure(bg="#4b5d67")

        for color in self.face_panels.keys():
            self.face_panels[color] = tk.Button(self.face_panel_container, text=color, bg="#000000",
                                                command=lambda col=color: self.rotateFace(col))

        # Solution Label
        self.solutionLabel = tk.Label(self, text="Sample Text", bg=self.window_bg, fg='white', font=('Helvetica', 30))

        # Buttons
        self.buttonFrame = tk.Frame(self, bg="#322f3d")
        self.addFaceBtn = tk.Button(self.buttonFrame, text="Add Face", height=2, width=15, command=self.updateFace)
        self.autoDetectBtn = tk.Button(self.buttonFrame, text="Auto Detect", height=2, width=15,
                                       command=self.toggleAutoDetect, bg='#fc6060')
        self.predictBtn = tk.Button(self.buttonFrame, borderwidth=0, border="0", bg=self.window_bg, activebackground=self.window_bg)
        self.predictBtn.config(image=self.predict_img)

        # Positioning Objects
        self.camera_feed_panel.grid(row=0, column=0, padx=(20, 0), pady=(20, 0), ipadx=0, ipady=0)
        self.place_face_panels()
        self.face_panel_container.grid(row=0, column=2, padx=50, pady=(20, 0))

        self.buttonFrame.grid(row=1, columnspan=3, padx=20, pady=(50, 0), ipadx=0, ipady=0)
        self.addFaceBtn.grid(row=0, column=1, padx=10, pady=0)
        self.predictBtn.grid(row=0, column=2, padx=10, pady=0, ipadx=0, ipady=0)
        self.autoDetectBtn.grid(row=0, column=3, padx=10, pady=0)

        self.solutionLabel.grid(row=2, columnspan=3, pady=(30, 0))

        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=0)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=1)
        self.grid_columnconfigure(3, weight=1)

    def place_face_panels(self):
        self.face_panels['BLUE'].grid(row=1, column=0, sticky='nsew', ipadx=0, padx=(30, 0))
        self.face_panels['RED'].grid(row=1, column=1, ipadx=0, ipady=0)
        self.face_panels['GREEN'].grid(row=1, column=2, ipadx=0, ipady=0)
        self.face_panels['ORANGE'].grid(row=1, column=3, ipadx=0, ipady=0, padx=(0, 30))
        self.face_panels['YELLOW'].grid(row=0, column=1, ipadx=0, ipady=0, pady=(30, 0))
        self.face_panels['WHITE'].grid(row=2, column=1, ipadx=0, ipady=0, pady=(0, 30))

        self.face_panel_container.grid_rowconfigure(0, weight=1)
        self.face_panel_container.grid_rowconfigure(1, weight=1)
        self.face_panel_container.grid_columnconfigure(0, weight=1)
        self.face_panel_container.grid_columnconfigure(1, weight=1)
        self.face_panel_container.grid_columnconfigure(2, weight=1)
        self.face_panel_container.grid_columnconfigure(3, weight=1)

    def showImage(self, read_img, frame=None, scale=1.0, x=None, y=None):
        if x is not None and y is not None:
            read_img = resize(read_img, x=x, y=y)

        read_img = rescale(read_img, scale)

        if frame is None:
            frame = self.camera_feed_panel
        image = self._prepareImg(read_img)
        # frame.configure(image=image)
        frame.create_image(0, 0, anchor='nw', image=image)
        frame.image = image

    def showAllFaces(self, face_img_dict):
        for col, face_panel in self.face_panels.items():
            cv_img = face_img_dict[col]
            image = self._prepareImg(cv_img)
            self.face_panels[col].configure(image=image)
            self.face_panels[col].image = image

    def showSolution(self):
        sol_string = self.cube_obj.getSolution()
        self.solutionLabel.config(text=sol_string)

    # Converts OpenCV Image to Tkinter image object
    def _prepareImg(self, read_img):
        read_img = cv.cvtColor(read_img, cv.COLOR_BGR2RGB)  # As cv use BGR but PIL uses RGB color space
        image = Image.fromarray(read_img)
        image = ImageTk.PhotoImage(image)
        return image

    # TODO Make event listener for the add face button
    def updateFace(self):
        # self.colorDetector.predictColor()
        colors = []
        for cnt in self.cubeResolver.final_contours:
            _, color = self.colorDetector.getSquareColor(self.cubeResolver.image, cnt)
            colors.append(color)
        face_col = self.cube_obj.addFace(colors)
        face_img = self.cube_obj.drawFace(face_col)
        self.cubeResolver.togglePause()

    def rotateFace(self, face_name):
        self.cube_obj.rotateFace(face_name)

    def toggleAutoDetect(self):
        self.doAutoDetect = not self.doAutoDetect
        if self.doAutoDetect:
            self.autoDetectBtn['bg'] = '#68fc60'
        else:
            self.autoDetectBtn['bg'] = '#fc6060'

    def run(self):
        while True:
            for cnt in self.cubeResolver.final_contours:
                cont_img, _ = self.colorDetector.getSquareColor(self.cubeResolver.image, cnt)
                self.showImage(cont_img, scale=0.77)
            if not self.cubeResolver.final_contours:
                contour_image = self.cubeResolver.getContourImage()
                self.showImage(contour_image, scale=0.77)

            self.cubeResolver.generateSquareContours()
            self.showAllFaces(self.cube_obj.getAllFaces(sideLen=30))
            # TODO Disabling enabling graphic image button
            if self.cube_obj.isSolveable():
                self.predictBtn['command'] = self.showSolution
                enable_btn(self.predictBtn, self.predict_img)
            else:
                self.predictBtn['command'] = None
                disable_btn(self.predictBtn, self.predict_img_disabled)

            if self.cubeResolver.isDetectionDone():
                self.addFaceBtn['state'] = 'normal'
                if self.doAutoDetect:
                    self.updateFace()
                if keyboard.is_pressed('r'):
                    self.cubeResolver.togglePause()
            else:
                self.addFaceBtn['state'] = 'disabled'

            self.update()


# TODO Show the contour color also in the live feed

if __name__ == '__main__':
    capture = cv.VideoCapture(0)
    cubeResolver = cuberesolver.CubeResolver(capture, SCALE=1, mode=1)
    colorDetector = colordetector.ColorDetector()
    cube = Cube.Cube()
    gui = App(colorDetector, cubeResolver, cube)

    gui.run()
