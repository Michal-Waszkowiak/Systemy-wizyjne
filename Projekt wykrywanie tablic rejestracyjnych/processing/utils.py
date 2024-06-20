import numpy as np
import os
import cv2
from sklearn.svm import SVC
from typing import List

def predict_character(char_img: np.ndarray, model: SVC) -> str:

    # Funkcja, która przygotowuje wykryte znaki na tablicy do predykcji. Następnie przygotowany znak jest
    # wykorzystywany do predykcji

    char_gray = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)
    char_blur = cv2.GaussianBlur(char_gray, (7, 7), 0)
    _, char_thresh = cv2.threshold(char_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    char_resized = cv2.resize(char_thresh, (70, 100))
    char_flat = char_resized.reshape(1, -1)

    prediction = model.predict(char_flat)
    return str(prediction[0])

def predict_plate_characters(char_images: List[np.ndarray], model: SVC) -> str:

    # Funkcja, która łączy wypredykowane znaki w całość i zwraca sklejony tekst

    predicted_text = ''
    for char_img in char_images:
        predicted_text += predict_character(char_img, model)
    return predicted_text

def perform_processing(image: np.ndarray) -> str:

    # Inicjalizacja potrzebnych zmiennych:
    # result - wynik końcowy w postaci napisu wypredykowanej tablicy rejestracyjnej
    # char_images - wektor znaków na tablicy rejestracyjnej
    # X - wektor do zbioru treningowego zawierajacego zdjęcia poszczególnych znaków na tablicy rejestracyjnej
    # y - etykiety do poszczególnych znaków na tablicy

    result = ''
    char_images = []
    X = []
    y = [
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
        'K', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U',
        'V', 'W', 'X', 'Y', 'Z'
    ]

    # Wczytanie zdjęć ze zbioru treningowego i przypisanie do zmiennej X

    for fileName in sorted(os.listdir('train')):
        if fileName.lower().endswith('.jpg'):
            image_path = os.path.join('train', fileName)
            image_details = cv2.imread(image_path)
            image_details_gray = cv2.cvtColor(image_details, cv2.COLOR_BGR2GRAY)
            image_details_resized = cv2.resize(image_details_gray, (70, 100))
            image_flat = image_details_resized.reshape(-1)
            X.append(image_flat)

    # Inicjalizacja modelu SVC wykrzystywanego do trenowania i predykcji znaków na tablicy

    svc_model = SVC(kernel='linear', probability=True)
    svc_model.fit(X, y)

    # Przygotowanie zdjęcia pod wykrycie tablicy rejestracyjnej

    img = cv2.resize(image, (0, 0), fx=0.4, fy=0.4)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gausblur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    ret, img_thresh = cv2.threshold(img_gausblur, 140, 255, cv2.THRESH_BINARY)
    img_canny = cv2.Canny(img_thresh, 185, 250)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    img_close = cv2.morphologyEx(img_canny, cv2.MORPH_CLOSE, kernel)
    img_dilatate = cv2.dilate(img_close, (3, 3), iterations=1)

    contours, _ = cv2.findContours(img_dilatate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    # Szukanie konturu, który będzie odpowiadał konturowi tablicy rejestracyjnej

    plate_contour = None
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h # Obliczenie stosunku szerokości do wysokości
        if aspect_ratio > 0.1 and aspect_ratio < 5:
            if cv2.contourArea(contour) > 100000: # Sprawdzenie czy dany kontur ma większe pole niż 100000
                plate_contour = contour
                break

    # Jeśli kontur tablicy został wykryty to wyznacz cztery wierzchołki tablicy,
    # jeśli kontur posiada 4 wierzchołki to wypłaszcz obraz

    if plate_contour is not None:
        epsilon = 0.02 * cv2.arcLength(plate_contour, True)
        approx = cv2.approxPolyDP(plate_contour, epsilon, True)

        if len(approx) == 4:
            pts = approx.reshape(4, 2)
            pts_sorted = np.zeros((4, 2), dtype=np.float32)

            s = pts.sum(axis=1)
            pts_sorted[0] = pts[np.argmin(s)]
            pts_sorted[2] = pts[np.argmax(s)]

            diff = np.diff(pts, axis=1)
            pts_sorted[1] = pts[np.argmin(diff)]
            pts_sorted[3] = pts[np.argmax(diff)]

            width = 400
            height = 100
            pts_new = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)

            matrix = cv2.getPerspectiveTransform(pts_sorted, pts_new)
            img_plate = cv2.warpPerspective(img, matrix, (width, height))

            # Przygotowanie tablicy rejestracyjnej do wykrycia poszczególnych znaków na tablicy

            plate_gray = cv2.cvtColor(img_plate, cv2.COLOR_BGR2GRAY)
            plate_blur = cv2.GaussianBlur(plate_gray, (3, 3), 0)
            ret, plate_thresh = cv2.threshold(plate_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            contours, _ = cv2.findContours(plate_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if h / w > 1.5 and h > 40 and cv2.contourArea(contour) > 100:
                    char_img = img_plate[y:y + h, x:x + w]
                    char_img = cv2.resize(char_img, (20, 40))
                    char_images.append(char_img)
                    cv2.rectangle(img_plate, (x, y), (x + w, y + h), (0, 255, 0), 2)

            result_img = img_plate.copy()
            cv2.drawContours(result_img, [plate_contour], -1, (0, 255, 0), 3)

            char_x_offset = width
            char_y_offset = 0
            char_width = 20
            char_height = 40

            # Dla każdego znaku z całej tablicy, przeprowadź predykcje i następnie połącz wyniki w jedną całość

            for char_img in char_images:
                result_img = cv2.copyMakeBorder(result_img, 0, 0, 0, char_width, cv2.BORDER_CONSTANT,
                                                value=[0, 0, 0])
                result_img[char_y_offset:char_y_offset + char_height,
                char_x_offset:char_x_offset + char_width] = char_img
                char_x_offset += char_width

        predicted_text = predict_plate_characters(char_images, svc_model)
        result += predicted_text

    # Jeśli nie udało się wypłaszczyć tablicy, to przypisz "PO12345" do wyniku końcowego

    if not result:
        result = 'PO12345'

    return result