import os
import cv2
import numpy as np

# Ścieżka do zdjęć
folder_path = "/home/michal/PycharmProjects/Systemy Wizyjne/LAB 01/lab01/Animacja"
delay = 100

images = []

# Wczytanie obrazów
for filename in os.listdir(folder_path):
    if filename.endswith(".jpg"):
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)
        if image is not None:
            images.append(image)

while True:

    key = cv2.waitKey(30)

    if key == ord('a'):
        delay = delay - 100
    if key == ord('s'):
        delay = delay + 100
    for image in images:

        cv2.imshow("Parabole", image)
        cv2.waitKey(delay)
        print(delay)

        cv2.waitKey(100)


cv2.destroyAllWindows()