import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread('zad_kółka_1.png')
copy = image.copy()
ref_point = []

def shape_selection(event, x, y, flags, param):
    global ref_point, cropped

    if event == cv2.EVENT_LBUTTONDOWN:
        ref_point = [(x,y)]

    elif event == cv2.EVENT_LBUTTONUP:
        ref_point.append((x,y))

        cv2.rectangle(image, ref_point[0], ref_point[1], (0,255,0), 2)
        cv2.imshow('image', image)

        cropped = image[min(ref_point[0][1], ref_point[1][1]):max(ref_point[0][1], ref_point[1][1]),
                  min(ref_point[0][0], ref_point[1][0]):max(ref_point[0][0], ref_point[1][0])]

def detect_edges(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    return edges
def TODO01():
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', shape_selection)


    while True:
        cv2.imshow('image', image)

        key_code = cv2.waitKey(10)
        if key_code == 27:  # Escape
            break

        elif key_code == ord('c'):  # Press 'c' to detect edges in the selected region
            if len(ref_point) == 2:
                edges = detect_edges(cropped)
                cv2.imshow('edges', edges)
                cv2.waitKey(0)
                cv2.destroyWindow('edges')
            elif key_code == ord('r'):
                cv2.imshow('image', copy)
        

    cv2.destroyAllWindows()