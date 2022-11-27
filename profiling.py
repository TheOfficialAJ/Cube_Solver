import cProfile
import pstats
import cv2 as cv

import colordetector
import cuberesolver
import snakeviz

img = cv.imread("Cube Images/cube1.jpg")
cubeResolver = cuberesolver.CubeResolver(img, SCALE=0.5, mode=0)
colorDetector = colordetector.ColorDetector()
cubeResolver.generateSquareContours()
with cProfile.Profile() as pr:
    for cnt in cubeResolver.final_contours:
        cont_img, _ = colorDetector.getSquareColor(cubeResolver.image, cnt)

stats = pstats.Stats(pr)
stats.sort_stats(pstats.SortKey.TIME)
stats.print_stats()
stats.dump_stats("cuberesolver_profile.prof")
