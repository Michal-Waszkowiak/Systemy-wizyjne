from __future__ import print_function
import argparse
import cv2
import numpy as np
import time


def empty_callback(value):
    pass

def save_frame(frame):
    filename = "alarm_frame.jpg"
    cv2.imwrite(filename, frame)
    print("Zapisano klatkę obrazu:", filename)

def zad01():
    gray_background = np.zeros((480,640,3), dtype=np.uint8)
    gray_current = np.zeros((480, 640, 3), dtype=np.uint8)
    foreground = np.zeros((480, 640, 3), dtype=np.uint8)
    thresh = np.zeros((480, 640, 3), dtype=np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    cv2.namedWindow("background image")
    cv2.namedWindow("current image")
    cv2.namedWindow("foreground image")
    cv2.createTrackbar('thresh','foreground image',0,255,empty_callback)

    cap = cv2.VideoCapture(0)

    while True:
        a = 0
        x = 2
        # Capture frame-by-frame
        ret, frame = cap.read()

        threshold_value = cv2.getTrackbarPos('thresh', 'foreground image')


        key_code = cv2.waitKey(100)
        if key_code == 27:  # Escape
            break
        elif key_code == ord('x'):  # a
            x = 1
            ret, frame = cap.read()
            gray_background = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


        elif key_code == ord('a'):  # x
            a = 1
            ret, frame = cap.read()
            gray_current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if a and x:
            foreground = cv2.absdiff(gray_current, gray_background, foreground)
            foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)

        ret, thresh = cv2.threshold(foreground, threshold_value, 255, cv2.THRESH_BINARY)

        cv2.imshow('background image', gray_background)
        cv2.imshow('current image', gray_current)
        cv2.imshow('foreground image', thresh)

def zad02():

    foreground = np.zeros((480,640,3), dtype=np.uint8)
    gray_background = np.zeros((480,640), dtype=np.uint8)
    gray_current = np.zeros((480,640), dtype=np.uint8)
    thresh = np.zeros((480,640), dtype=np.uint8)
    kernel = np.ones((5, 5), np.uint8)

    cv2.namedWindow("background image")
    cv2.namedWindow("current image")
    cv2.namedWindow("foreground image")
    cv2.createTrackbar('thresh','foreground image',0,255, lambda x: None)

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        rows, cols, _ = frame.shape
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if not np.any(gray_background):
            gray_background = gray_frame.copy()
            continue
        for i in range(rows):
            for j in range(cols):
                if gray_background[i,j] < gray_frame[i,j]:
                    gray_background[i,j] =+ 1
                elif gray_background[i,j] > gray_frame[i,j]:
                    gray_background[i,j] =- 1

        foreground = cv2.absdiff(gray_background, gray_frame, foreground)
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)

        threshold_value = cv2.getTrackbarPos('thresh', 'foreground image')
        ret, thresh = cv2.threshold(foreground, threshold_value, 255, cv2.THRESH_BINARY)

        cv2.imshow('background image', gray_background)
        cv2.imshow('current image', gray_frame)
        cv2.imshow('foreground image', thresh)

        key_code = cv2.waitKey(30)
        if key_code == 27:  # Escape
            break

    cap.release()
    cv2.destroyAllWindows()

def zad03():
    # Inicjalizacja modelu tła jako pierwsza klatka obrazu
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    gray_background = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Utworzenie okna do wyświetlania obrazów
    cv2.namedWindow("background image")
    cv2.namedWindow("current image")
    cv2.namedWindow("foreground image")
    cv2.createTrackbar('thresh', 'foreground image', 0, 255, lambda x: None)

    # Pętla główna
    while True:
        # Pobranie obrazu wejściowego
        ret, frame = cap.read()
        gray_current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Aktualizacja modelu tła na podstawie obrazu wejściowego
        mask_less = gray_background < gray_current
        mask_greater = gray_background > gray_current
        gray_background[mask_less] += 1
        gray_background[mask_greater] -= 1

        # Obliczenie różnicy między obrazem wejściowym a modelem tła
        foreground = cv2.absdiff(gray_background, gray_current)

        # Progowanie różnicy
        threshold_value = cv2.getTrackbarPos('thresh', 'foreground image')
        ret, thresh = cv2.threshold(foreground, threshold_value, 255, cv2.THRESH_BINARY)

        # Wyświetlenie obrazów
        cv2.imshow('background image', gray_background)
        cv2.imshow('current image', gray_current)
        cv2.imshow('foreground image', thresh)

        # Wyjście z pętli po naciśnięciu klawisza Escape
        key_code = cv2.waitKey(30)
        if key_code == 27:
            break

    # Zwolnienie zasobów i zamknięcie okien
    cap.release()
    cv2.destroyAllWindows()

def zad04():
    parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
     OpenCV. You can process both videos and images.')
    parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='vtest.avi')
    parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
    args = parser.parse_args()

    if args.algo == 'MOG2':
        backSub = cv2.createBackgroundSubtractorMOG2()
    else:
        backSub = cv2.createBackgroundSubtractorKNN()

    capture = cv2.VideoCapture(cv2.samples.findFileOrKeep(args.input))
    if not capture.isOpened():
        print('Unable to open: ' + args.input)
        exit(0)

    while True:
        ret, frame = capture.read()
        if frame is None:
            break

        fgMask = backSub.apply(frame)

        cv2.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
        cv2.putText(frame, str(capture.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        cv2.imshow('Frame', frame)
        cv2.imshow('FG Mask', fgMask)

        keyboard = cv2.waitKey(30)
        if keyboard == 'q' or keyboard == 27:
            break

def zad05():
    # Inicjalizacja modelu tła jako pierwsza klatka obrazu
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    gray_background = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Utworzenie okna do wyświetlania obrazów
    cv2.namedWindow("background image")
    cv2.namedWindow("current image")
    cv2.namedWindow("foreground image")
    cv2.createTrackbar('thresh', 'foreground image', 90, 255, empty_callback)

    # Ustawienie progu aktywacji alarmu (ilość zmienionych pikseli)
    alarm_threshold = 50000

    # Pętla główna
    while True:
        # Pobranie obrazu wejściowego
        ret, frame = cap.read()
        gray_current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Aktualizacja modelu tła na podstawie obrazu wejściowego
        mask_less = gray_background < gray_current
        mask_greater = gray_background > gray_current
        gray_background[mask_less] += 1
        gray_background[mask_greater] -= 1

        # Obliczenie różnicy między obrazem wejściowym a modelem tła
        foreground = cv2.absdiff(gray_background, gray_current)

        # Progowanie różnicy
        threshold_value = cv2.getTrackbarPos('thresh', 'foreground image')
        ret, thresh = cv2.threshold(foreground, threshold_value, 255, cv2.THRESH_BINARY)

        # Znalezienie konturów zmienionych obszarów
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Narysowanie ramki dookoła pikseli oznaczonych jako zmienione
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        num_changed_pixels = cv2.countNonZero(thresh)

        # Wyświetlenie obrazów
        cv2.imshow('background image', gray_background)
        cv2.imshow('current image', gray_current)
        cv2.imshow('foreground image', thresh)

        # cv2.imshow('With Rectangle', frame)

        # Sprawdzenie warunku uruchomienia alarmu
        if num_changed_pixels > alarm_threshold:
            print("Alarm aktywowany! Zmienione piksele:", num_changed_pixels)
            # save_frame(frame)

        # Wyjście z pętli po naciśnięciu klawisza Escape
        key_code = cv2.waitKey(30)
        if key_code == 27:
            break

    # Zwolnienie zasobów i zamknięcie okien
    cap.release()
    cv2.destroyAllWindows()

def zaddom1():
    previousframe = np.zeros((480, 640, 3), dtype=np.uint8)
    current_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    next_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    foreground = np.zeros((480, 640, 3), dtype=np.uint8)
    gray_background = np.zeros((480, 640), dtype=np.uint8)
    gray_current = np.zeros((480, 640), dtype=np.uint8)
    thresh = np.zeros((480, 640), dtype=np.uint8)
    kernel = np.ones((5, 5), np.uint8)

    cv2.namedWindow("background image")
    cv2.namedWindow("current image")
    cv2.namedWindow("foreground image")
    cv2.createTrackbar('thresh', 'foreground image', 0, 255, lambda x: None)

    cap = cv2.VideoCapture(0)

    lower_color = np.array([40, 50, 50])
    upper_color = np.array([80, 255, 255])

    while True:
        ret, frame = cap.read()
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        previous_frame = current_frame.copy()
        current_frame = next_frame.copy()
        next_frame = hsv_frame.copy()

        if not np.any(current_frame):
            continue

        previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
        current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        diff1 = cv2.absdiff(next_frame, current_frame)
        diff2 = cv2.absdiff(next_frame, previous_frame)
        foreground = cv2.bitwise_or(diff1, diff2)
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)

        # Filtracja koloru
        mask = cv2.inRange(hsv_frame, lower_color, upper_color)

        threshold_value = cv2.getTrackbarPos('thresh', 'foreground image')
        ret, thresh = cv2.threshold(mask, threshold_value, 255, cv2.THRESH_BINARY)

        cv2.imshow('background image', current_frame)
        cv2.imshow('current image', next_frame)
        cv2.imshow('foreground image', thresh)

        key_code = cv2.waitKey(30)
        if key_code == 27:  # Escape
            break

    cap.release()
    cv2.destroyAllWindows()

    import time

def ex_1():

    cv2.namedWindow('current_frame')
    cv2.namedWindow('background')
    cv2.namedWindow('foreground')

    cv2.createTrackbar('threshold', 'current_frame', 20, 255, empty_callback)

    cap = cv2.VideoCapture(1)

    img_gray = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2GRAY)
    img_current = np.copy(img_gray)
    img_background = np.copy(img_gray)
    img_foreground = np.copy(img_gray)

    backSub = cv2.createBackgroundSubtractorMOG2()
    # backSub = cv2.createBackgroundSubtractorKNN()

    key = ord(' ')
    while key != ord('q'):
        _, frame = cap.read()
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        fgMask = backSub.apply(frame)

        # background update
        # if key == ord('a'):
        img_background = np.copy(img_current)
        img_background[img_background < img_current] += 1
        img_background[img_background > img_current] -= 1

        # elif key == ord('x'):
        img_current = np.copy(img_gray)

        img_diff = cv2.absdiff(img_background, img_current)
        kernel = np.ones((5, 5), np.uint8)
        img_closed = cv2.morphologyEx(img_diff, cv2.MORPH_OPEN, kernel)

        t = cv2.getTrackbarPos('threshold', 'current_frame')
        _, img_thresholded = cv2.threshold(img_closed, t, 255, cv2.THRESH_BINARY)

        cv2.imshow('current_frame', img_current)
        cv2.imshow('background', img_background)
        cv2.imshow('foreground', img_thresholded)
        cv2.imshow('fgMask', fgMask)

        key = cv2.waitKey(20)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # zad01()
    # zad02()
    # zad03()
    # zad04()
    # zad05()
    # zaddom1()
    ex_1()
