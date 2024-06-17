import cv2
import numpy as np

def empty_callback(value):
    pass

image_Base = cv2.imread('BRG.jpg')
image_ROI = np.zeros((260,554), dtype =np.uint8)
image_Thresh = np.zeros((260,554), dtype =np.uint8)
image_Final = np.zeros((260,554,3), dtype =np.uint8)

#blue, green, red = cv2.split(image_Base)

image_Base_1 = image_Base[:130, :277]
image_Base_2 = image_Base[:130, 277:]
image_Base_3 = image_Base[130:, :277]
image_Base_4 = image_Base[130:, 277:]

gray = cv2.cvtColor(image_Base, cv2.COLOR_BGR2GRAY)
red_R, red_B, red_G = cv2.split(image_Base_2)
green_R, green_B, green_G = cv2.split(image_Base_3)
blue_R, blue_B, blue_G = cv2.split(image_Base_4)

# image_BLue = cv2.merge((blue,green*0,red*0))
# image_Green = cv2.merge((blue*0,green,red*0))
# image_Red = cv2.merge((blue*0,green*0,red))

gray_quarter = gray[:130, :277]

image_ROI[:130, :277] = gray_quarter
image_ROI[:130, 277:] = red_R
image_ROI[130:, :277] = green_G
image_ROI[130:, 277:] = blue_B

# top_half = cv2.hconcat([gray_quarter,red_quarter])
# bottom_half = cv2.hconcat([green_quarter,blue_quarter])
# combined_image = cv2.vconcat([top_half,bottom_half])

# Utwórz okno
cv2.namedWindow('Base')
cv2.namedWindow('Splitted')
cv2.namedWindow('Threshold')
cv2.namedWindow('Back to normal')

# Utwórz trackbar do regulacji progu
cv2.createTrackbar('Gray', 'Splitted', 0, 255, empty_callback)
cv2.createTrackbar('Red', 'Splitted', 0, 255, empty_callback)
cv2.createTrackbar('Green', 'Splitted', 0, 255, empty_callback)
cv2.createTrackbar('Blue', 'Splitted', 0, 255, empty_callback)

while True:
    # Pobierz wartość progu z trackbara
    threshold_value_gray = cv2.getTrackbarPos('Gray', 'Splitted')
    threshold_value_red = cv2.getTrackbarPos('Red', 'Splitted')
    threshold_value_green = cv2.getTrackbarPos('Green', 'Splitted')
    threshold_value_blue = cv2.getTrackbarPos('Blue', 'Splitted')

    # Użyj wartości progu do binaryzacji
    ret, thresh_gray = cv2.threshold(gray_quarter, threshold_value_gray, 255, cv2.THRESH_BINARY)
    ret, thresh_red = cv2.threshold(image_ROI[:130, 277:], threshold_value_red, 255, cv2.THRESH_BINARY)
    ret, thresh_green = cv2.threshold(image_ROI[130:, :277], threshold_value_green, 255, cv2.THRESH_BINARY)
    ret, thresh_blue = cv2.threshold(image_ROI[130:, 277:], threshold_value_blue, 255, cv2.THRESH_BINARY)

    image_Thresh[:130, :277] = thresh_gray
    image_Thresh[:130, 277:] = thresh_red
    image_Thresh[130:, :277] = thresh_green
    image_Thresh[130:, 277:] = thresh_blue

    # thresh_top_half = cv2.hconcat([thresh_gray, thresh_red])
    # thresh_bottom_half = cv2.hconcat([thresh_green, thresh_blue])
    # thresh_combined_image = cv2.vconcat([thresh_top_half, thresh_bottom_half])

    final_gray = image_Base_1 * cv2.merge([thresh_gray, thresh_gray, thresh_gray]) * -1
    final_red = cv2.merge((blue_R, green_R,image_Base_2[:, :, 0] * -thresh_red ))
    final_green = cv2.merge((red_G, image_Base_3[:, :, 1] * -thresh_green, blue_G))
    final_blue = cv2.merge((image_Base_4[:, :, 2] * -thresh_blue, green_B, red_B))

    image_Final[:130, :277] = final_gray
    image_Final[:130, 277:] = final_red
    image_Final[130:, :277] = final_green
    image_Final[130:, 277:] = final_blue

    # Wyświetl obrazy w odpowiednich oknach
    # cv2.imshow('Base',image_Base)
    cv2.imshow('Splitted', image_ROI)
    cv2.imshow('Threshold', image_Thresh)
    cv2.imshow('Back to normal', image_Final)

    key_code = cv2.waitKey(10)
    if key_code == 27:  # Escape
        break

