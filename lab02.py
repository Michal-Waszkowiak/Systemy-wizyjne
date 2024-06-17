import cv2
import sys
from matplotlib import pyplot as plt
import numpy as np


def empty_callback(value):
    pass
# def empty_callback_green(value):
#     print(f'Trackbar reporting for duty with Green value: {value}')
#     pass
# def empty_callback_blue(value):
#     print(f'Trackbar reporting for duty with Blue value: {value}')
#     pass
# def empty_callback(value):
#     print(f'Trackbar reporting for duty with value: {value}')
#     pass
#
# # create a black image, a window
# img = np.zeros((300, 512, 3), dtype=np.uint8)
# cv2.namedWindow('image')
#
# # create trackbars for color change
# cv2.createTrackbar('R', 'image', 0, 255, empty_callback_red)
# cv2.createTrackbar('G', 'image', 0, 255, empty_callback_green)
# cv2.createTrackbar('B', 'image', 0, 255, empty_callback_blue)
#
# # create switch for ON/OFF functionality
# switch_trackbar_name = '0 : OFF \n1 : ON'
# cv2.createTrackbar(switch_trackbar_name, 'image', 0, 1, empty_callback)
#
# while True:
#     cv2.imshow('image', img)
#
#     # sleep for 10 ms waiting for user to press some key, return -1 on timeout
#     key_code = cv2.waitKey(10)
#     if key_code == 27:
#         # escape key pressed
#         break
#
#     # get current positions of four trackbars
#     r = cv2.getTrackbarPos('R', 'image')
#     g = cv2.getTrackbarPos('G', 'image')
#     b = cv2.getTrackbarPos('B', 'image')
#     s = cv2.getTrackbarPos(switch_trackbar_name, 'image')
#
#     if s == 0:
#         # assign zeros to all pixels
#         img[:] = 0
#     else:
#         # assign the same BGR color to all pixels
#         img[:] = [b, g, r]
#
# # closes all windows (usually optional as the script ends anyway)
# cv2.destroyAllWindows()


###########################################################################################

# def empty_callback(value):
#     pass
#
# img = cv2.imread("tecza.png")
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# # Utwórz okno
# cv2.namedWindow('image')
#
# # Utwórz trackbar do regulacji progu
# cv2.createTrackbar('Threshold Value', 'image', 0, 255, empty_callback)
#
# # Utwórz trackbar do wyboru typu progowania
# cv2.createTrackbar('Threshold Type', 'image', 0, 4, empty_callback)
#
# if img is None:
#     sys.exit("Could not read the image.")
#
# while True:
#     # Pobierz wartość progu z trackbara
#     threshold_value = cv2.getTrackbarPos('Threshold Value', 'image')
#
#     # Pobierz typ progowania z trackbara
#     threshold_type = cv2.getTrackbarPos('Threshold Type', 'image')
#
#     # Wybierz odpowiedni typ progowania
#     if threshold_type == 0:
#         threshold_type = cv2.THRESH_BINARY
#     elif threshold_type == 1:
#         threshold_type = cv2.THRESH_BINARY_INV
#     elif threshold_type == 2:
#         threshold_type = cv2.THRESH_TRUNC
#     elif threshold_type == 3:
#         threshold_type = cv2.THRESH_TOZERO
#     elif threshold_type == 4:
#         threshold_type = cv2.THRESH_TOZERO_INV
#
#     # Użyj wartości progu do binaryzacji
#     ret, thresh1 = cv2.threshold(img_gray, threshold_value, 255, threshold_type)
#
#     cv2.imshow('image', thresh1)
#
#     # Czekaj na naciśnięcie klawisza przez użytkownika
#     key_code = cv2.waitKey(10)
#     if key_code == 27:  # Escape
#         break
#
# cv2.destroyAllWindows()

############################################################################################

img = cv2.imread("qr.jpg")
if img is None:
    sys.exit("Could not read the image.")

scale_factor = 2.75


methods = [
    cv2.INTER_LINEAR,
    cv2.INTER_NEAREST,
    cv2.INTER_AREA,
    cv2.INTER_LANCZOS4]


titles=['Inter_lin','Inter_near','Inter_area','Inter_lan']
images = []
for i in range(4):
    img_scaled = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=methods[i])
    images.append(img_scaled)
    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray',vmin=0,vmax=255)
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()

