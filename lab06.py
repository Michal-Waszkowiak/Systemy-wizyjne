import cv2
import numpy as np

from matplotlib import pyplot as plt


def zad01():
    image = cv2.imread('not_bad.jpg')
    # cv2.imshow('Oryginal',image)
    image_resize = cv2.resize(image, (1280,720))
    rows, cols, ch = image_resize.shape
    # cv2.imshow('Resize', image_resize)
    image_gray = cv2.cvtColor(image_resize,cv2.COLOR_BGR2GRAY)
    ret, thresh_gray = cv2.threshold(image_gray, 55, 255, cv2.THRESH_BINARY)
    # cv2.imshow('Threshold', thresh_gray)
    kernel = np.ones((5, 5), np.uint8)
    image_dilate = cv2.dilate(thresh_gray, kernel, iterations=1)
    # cv2.imshow('Dilate', image_dilate)

    contours, hierarchy = cv2.findContours(image_dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt1 = contours[1]
    cnt2 = contours[2]
    cnt3 = contours[3]
    cnt4 = contours[4]
    cv2.drawContours(image_resize, [cnt1,cnt2,cnt3,cnt4], -1, (0, 255, 0), 3)


    M1 = cv2.moments(cnt1)
    cx1 = int(M1['m10'] / M1['m00'])
    cy1 = int(M1['m01'] / M1['m00'])
    area1 = cv2.contourArea(cnt1)
    cv2.circle(image_resize,(cx1,cy1), 1, (255,255,255), 2)

    M2 = cv2.moments(cnt2)
    cx2 = int(M2['m10'] / M2['m00'])
    cy2 = int(M2['m01'] / M2['m00'])
    area2 = cv2.contourArea(cnt2)
    cv2.circle(image_resize, (cx2, cy2), 1, (255, 255, 255), 2)

    M3 = cv2.moments(cnt3)
    cx3 = int(M3['m10'] / M3['m00'])
    cy3 = int(M3['m01'] / M3['m00'])
    area3 = cv2.contourArea(cnt3)
    cv2.circle(image_resize, (cx3, cy3), 1, (255, 255, 255), 2)

    M4 = cv2.moments(cnt4)
    cx4 = int(M4['m10'] / M4['m00'])
    cy4 = int(M4['m01'] / M4['m00'])
    area4 = cv2.contourArea(cnt4)
    cv2.circle(image_resize, (cx4, cy4), 1, (255, 255, 255), 2)

    cv2.imshow('Contours',image_resize)

    pts1 = np.float32([[cx1,cy1], [cx2, cy2], [cx3, cy3], [cx4,cy4]])
    pts2 = np.float32([[1280, 720], [0, 720], [1280, 0], [0,0]])

    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(image_resize, M, (cols, rows))
    cv2.imshow('Perspective', dst)

    print(image_resize.shape)

def zad02():
    image = cv2.imread('dre.png')
    img2 = image.copy()
    # cv2.imshow('Oryginal',image)
    image_template = image[150:200, 150:220]
    # cv2.imshow('Template', image_template)
    image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # cv2.imshow('Oryginal Gray', image_gray)
    template_gray = cv2.cvtColor(image_template,cv2.COLOR_BGR2GRAY)
    # cv2.imshow('Template Gray', template_gray)
    w, h = template_gray.shape

    res = cv2.matchTemplate(image_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (255, 0, 0), 1)

    cv2.imshow('Image', image)






if __name__ == '__main__':
    # zad01()
    zad02()

    cv2.waitKey(0)
    cv2.destroyAllWindows()