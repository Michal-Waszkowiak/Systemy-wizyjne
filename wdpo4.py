import cv2 as cv
import numpy as np

#Wczytaj obraz o wymairach axa
image = cv.imread('One_Ringjpg.jpg')
h, w, _ = image.shape

def empty_callback(value):
    pass
cv.imshow('Base Image', image)
cv.namedWindow('Rotated Image')
cv.createTrackbar('Rotate', 'Base Image', 0, 360,empty_callback)

pts = np.float32([[0,w], [0,0], [h, 0]])

while True:
    angle_rotate = cv.getTrackbarPos('Rotate', 'Base Image')

    value_rotate = int((angle_rotate/90 * h) % h)

    if 0 <= angle_rotate < 90 or angle_rotate == 360:
        pts_changed = [ [0, w - value_rotate], [value_rotate, 0], [h, value_rotate] ]
    elif 90 <= angle_rotate < 180:
        pts_changed = [ [value_rotate, 0], [h, value_rotate], [h - value_rotate, w] ]
    elif 180 <= angle_rotate < 270:
        pts_changed = [ [h, value_rotate], [h - value_rotate, w], [0, w - value_rotate] ]
    else:
        pts_changed = [ [h - value_rotate, w], [0, w - value_rotate], [value_rotate, 0]]

    pts_changed = np.float32(pts_changed)
    M = cv.getAffineTransform(pts,pts_changed)
    dst = cv.warpAffine(image,M,(w,h))

    cv.imshow('Rotated Image', dst)

    key_code = cv.waitKey(10)
    if key_code == 27:
        break

cv.destroyAllWindows()