import os
import cv2

# Ścieżka do folderu zawierającego zdjęcia
folder_path = "/home/michal/PycharmProjects/Systemy Wizyjne/LAB 01/lab01/Animacja"

# Lista do przechowywania wczytanych obrazów
images = []

# Pętla po wszystkich plikach w folderze
for filename in os.listdir(folder_path):
    # Sprawdzenie, czy plik ma rozszerzenie obrazu
    if filename.endswith(('.jpg')):
        # Pełna ścieżka do pliku
        image_path = os.path.join(folder_path, filename)
        # Wczytanie obrazu i dodanie go do listy
        image = cv2.imread(image_path)
        if image is not None:
            images.append(image)

def nonStopImage(idx,delay):
    while True:

        cv2.imshow("Parabole tańczą", images[idx])
        idx = (idx + 1) % len(images)
        #print(delay)
        cv2.waitKey(delay)

        key_code_restart = cv2.waitKey(10)
        if key_code_restart == 27:  # Escape
            print('Reset')
            break

def TODO1():
    delay = 500
    cur_idx = 0
    while True:
        cv2.imshow("Parabole tańczą", images[cur_idx])
        key_code = cv2.waitKey(delay)

        if key_code == ord('a'):
            nonStopImage(cur_idx,delay)

        elif key_code == ord('w'):
            cur_idx = (cur_idx + 1) % len(images)

        elif key_code == ord('q'):
            cur_idx = (cur_idx - 1) % len(images)

        elif key_code == ord('z'):
            if delay >= 200:
                delay = delay - 100
                print('Current delay ' + str(round(delay, 1)) + ' ms')

        elif key_code == ord('x'):
            delay = delay + 100
            print('Current delay ' + str(round(delay, 1)) + ' ms')

        elif key_code == ord('k'):
            cv2.destroyAllWindows()
            break

if __name__ == '__main__':
    TODO1()