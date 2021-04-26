import tkinter as tk
from PIL import Image, ImageTk
import cv2 as cv
from resize import resize


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.geometry('1080x720')
        self.activeFrame = None
        self.frames = {}
        self.camera_feed_panel = tk.Label(self)

        self.face_panel_container = tk.LabelFrame(self)
        self.face_panels = {'RED': None, 'BLUE': None, 'GREEN': None, 'YELLOW': None, 'WHITE': None, 'ORANGE': None}
        for color in self.face_panels.keys():
            self.face_panels[color] = tk.Label(self.face_panel_container, text=color)

        self.camera_feed_panel.grid(row=0, column=0, padx=10, pady=10)
        self.pack_labels()

        self.face_panel_container.grid(row=0, column=1, padx=20, pady=20)

    def pack_labels(self):
        self.face_panels['BLUE'].grid(row=1, column=0)
        self.face_panels['RED'].grid(row=1, column=1)
        self.face_panels['GREEN'].grid(row=1, column=2)
        self.face_panels['ORANGE'].grid(row=1, column=3)
        self.face_panels['YELLOW'].grid(row=0, column=1)
        self.face_panels['WHITE'].grid(row=2, column=1)

    def showImage(self, read_img, frame=None, scale=1):
        read_img = resize(read_img, scale)
        if frame is None:
            frame = self.camera_feed_panel
        image = self._prepareImg(read_img)
        frame.configure(image=image)
        frame.image = image

    def showAllFaces(self, faces_dict):
        for col, face_panel in self.face_panels.items():
            cv_img = faces_dict[col]
            image = self._prepareImg(cv_img)
            self.face_panels[col].configure(image=image)
            self.face_panels[col].image = image

    # Converts OpenCV Image to Tkinter image object
    def _prepareImg(self, read_img):
        read_img = cv.cvtColor(read_img, cv.COLOR_BGR2RGB)  # As cv use BGR but PIL uses RGB color space
        image = Image.fromarray(read_img)
        image = ImageTk.PhotoImage(image)
        return image
