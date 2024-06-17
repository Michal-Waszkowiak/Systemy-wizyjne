import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def TODO01():

    while True:


        key_code = cv.waitKey(10)
        if key_code == 27:  # Escape
            break

def TODO02():
    image = cv.imread('cars.png')
    image_grey = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    ret, thresh = cv.threshold(image_grey,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

    kernel = np.ones((3,3),np.uint8)
    opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel,iterations=2)

    sure_background = cv.dilate(opening,kernel,iterations=3)

    distance_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
    ret, sure_foreground = cv.threshold(distance_transform, 0.1*distance_transform.max(),255,0)

    sure_foreground = np.uint8(sure_foreground)
    unknown = cv.subtract(sure_background,sure_foreground)

    ret, markers = cv.connectedComponents(sure_foreground)

    markers = markers + 1

    markers[unknown==255] = 0

    markers = cv.watershed(image, markers)
    image[markers == -1] = [255, 0, 0]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(markers, cmap="tab20b")
    ax.axis('off')
    plt.show()

    labels = np.unique(markers)

    coins = []
    for label in labels[2:]:
        # Create a binary image in which only the area of the label is in the foreground
        # and the rest of the image is in the background
        target = np.where(markers == label, 255, 0).astype(np.uint8)

        # Perform contour extraction on the created binary image
        contours, hierarchy = cv.findContours(
            target, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )
        coins.append(contours[0])

    # Draw the outline
    image = cv.drawContours(image, coins, -1, color=(0, 23, 223), thickness=2)



    while True:
        cv.imshow('oryginal',image)
        # cv.imshow('Thresh',thresh)
        # cv.imshow('Open',opening)
        # cv.imshow('Sure_back',sure_background)
        # cv.imshow('Sure_fore',sure_foreground)
        # cv.imshow('Unknown',unknown)
        # cv.imshow('Markers', markers)


        key_code = cv.waitKey(10)
        if key_code == 27:  # Escape
            break
def TODO03():

    image = cv.imread('tumor.jpg')
    mask = np.zeros(image.shape[:2], np.uint8)
    backgroundModel = np.zeros((1, 65), np.float64)
    foregroundModel = np.zeros((1, 65), np.float64)
    rectangle = (330, 620, 150, 150)
    cv.grabCut(image, mask, rectangle,backgroundModel, foregroundModel,3, cv.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    image_segmented = image * mask2[:, :, np.newaxis]

    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Segmented Image')
    plt.imshow(cv.cvtColor(image_segmented, cv.COLOR_BGR2RGB))
    plt.axis('off')

    plt.show()



if __name__ == '__main__':
    # TODO01()
    # TODO02()
    TODO03()