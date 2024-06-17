import cv2 as cv
import numpy as np
import time
from matplotlib import pyplot as plt


lenna_noise = cv.imread('lenna_noise.bmp')
lenna_sap = cv.imread('lenna_salt_and_pepper.bmp')
black_white = cv.imread('black_white.jpg', cv.IMREAD_GRAYSCALE)
brg_image = cv.imread('BRG.jpg',cv.IMREAD_GRAYSCALE)
square_image = cv.imread('kwadrat.jpg', cv.IMREAD_GRAYSCALE)

def blur_median_gaus(image,image2, filter_size):

    blur = cv.blur(image,(filter_size,filter_size))
    gausian = cv.GaussianBlur(image, (filter_size, filter_size), 0)
    median = cv.medianBlur(image, filter_size)
    blur2 = cv.blur(image2, (filter_size, filter_size))
    gausian2 = cv.GaussianBlur(image2, (filter_size, filter_size), 0)
    median2 = cv.medianBlur(image2, filter_size)

    plt.subplot(141),plt.imshow(image),plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(142),plt.imshow(blur),plt.title('Blurred')
    plt.xticks([]), plt.yticks([])
    plt.subplot(143), plt.imshow(gausian), plt.title('Gausian')
    plt.xticks([]), plt.yticks([])
    plt.subplot(144), plt.imshow(median), plt.title('Median')
    plt.xticks([]), plt.yticks([])
    plt.subplot(441), plt.imshow(image2), plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(442), plt.imshow(blur2), plt.title('Blurred')
    plt.xticks([]), plt.yticks([])
    plt.subplot(443), plt.imshow(gausian2), plt.title('Gausian')
    plt.xticks([]), plt.yticks([])
    plt.subplot(444), plt.imshow(median2), plt.title('Median')
    plt.xticks([]), plt.yticks([])
    plt.show()

def on_trackbar(val):
    filter_size = val * 2 + 1
    blur_median_gaus(lenna_noise,lenna_sap,filter_size)

def operacje_morfologiczne(image):
    kernel = np.ones((9,9),np.uint8)
    erosion = cv.erode(image,kernel,iterations=1)
    dilation = cv.dilate(image, kernel, iterations=1)
    opening = cv.morphologyEx(image, cv.MORPH_OPEN, kernel)
    closing = cv.morphologyEx(image, cv.MORPH_CLOSE, kernel)
    gradient = cv.morphologyEx(image, cv.MORPH_GRADIENT, kernel)
    tophat = cv.morphologyEx(image, cv.MORPH_TOPHAT, kernel)
    blackhat = cv.morphologyEx(image, cv.MORPH_BLACKHAT, kernel)
    cv.imshow('Original', image)
    # cv.imshow('Erosion',erosion)
    # cv.imshow('Dilation', dilation)
    # cv.imshow('Opening',opening)
    # cv.imshow('Closing', closing)
    # cv.imshow('Gradient', gradient)
    # cv.imshow('Top Hat', tophat)
    cv.imshow('Black Hat', blackhat)

def skanowanie_obrazu(image):
    cv.imshow('Original',image)
    start_time_custom = time.time()

    smoothed_image = np.zeros_like(image)
    height, width = image.shape

    for y in range(height):
        for x in range(0,width,3):
            image[y,x] = 255

    for y in range(1, height-1):
        for x in range(1, width-1):
            sum_pixels = (
                np.sum(image[y-1:y+2, x-1:x+2]) +
                image[y-1,x] + image[y+1,x] +
                image[y,x-1] + image[y,x+1]
            )

            smoothed_pixel_value = sum_pixels // 9
            smoothed_image[y,x] = smoothed_pixel_value

    end_time_custom = time.time()
    custom_duration = end_time_custom - start_time_custom

    #Blur fun
    start_time_blur = time.time()
    smoothed_image_blur = cv.blur(image,(3,3))
    end_time_blur = time.time()
    blur_duration = end_time_blur - start_time_blur

    #filter2D
    start_time_filter2D = time.time()
    smoothed_img_filter2D = cv.filter2D(image, -1, (3,3))
    end_time_filter2D = time.time()
    filter2D_duration = end_time_filter2D - start_time_filter2D

    cv.imshow('Changed',image)
    cv.imshow('Smoothed',smoothed_image)
    cv.imshow('Smoothed blur',smoothed_image_blur)
    cv.imshow('Smoothed Filter2D',smoothed_img_filter2D)

    # Wyświetl czas wykonania dla obu metod
    print("Czas wykonania własnej implementacji: {:.5f} sekund".format(custom_duration))
    print("Czas wykonania funkcji cv2.blur(): {:.5f} sekund".format(blur_duration))
    print("Czas wykonania funkcji cv.filter2D: {:.5f} sekund".format(filter2D_duration))

def filtr_Kuwahary(image,window_size):
    cv.imshow('Original',image)

    # Pobierz wymiary obrazu
    height, width = image.shape[:2]

    # Utwórz obraz wynikowy
    filtered_image = np.zeros_like(image)

    # Współrzędne środka okna
    half_window = window_size // 2

    for y in range(half_window, height - half_window):
        for x in range(half_window, width - half_window):
            # Podziel okno na cztery części
            regions = [
                image[y - half_window: y + half_window, x - half_window: x + half_window],
                image[y - half_window: y + half_window, x: x + 2*half_window + 1],
                image[y: y + 2*half_window + 1, x - half_window: x + half_window],
                image[y: y + 2*half_window + 1, x: x + 2*half_window + 1]
            ]

            # Inicjalizuj listę średnich i odchyleń standardowych
            means = []
            std_devs = []

            # Oblicz średnie i odchylenia standardowe dla każdej części
            for region in regions:
                mean, std_dev = cv.meanStdDev(region)
                means.append(mean)
                std_devs.append(std_dev)

            # Znajdź część z najmniejszym odchyleniem standardowym
            min_std_dev_index = np.argmin(std_devs)

            # Przypisz średnią z części o najmniejszym odchyleniu standardowym do piksela wynikowego
            filtered_image[y, x] = means[min_std_dev_index]

        cv.imshow('Kuwahara Filtered Image',filtered_image)

def wdpo3():
    dokument_image = cv.imread('dokument.png')
    dokument_image_gray = cv.cvtColor(dokument_image,cv.COLOR_BGR2GRAY)
    kernel = np.ones((2, 2), np.uint8)
    assert dokument_image is not None, "file could not be read, check with os.path.exists()"
    #cv.imshow('Original', dokument_image)

    dokument_resize = cv.resize(dokument_image_gray,None, fx=1,fy=1)
    # cv.imshow('Resize', dokument_resize)

    # dokument_blur = cv.blur(dokument_resize,(5,5))
    # cv.imshow('Blur',dokument_blur)


    dokument_gauss = cv.GaussianBlur(dokument_resize, (9,9), 0)
    # cv.imshow('Gauss', dokument_gauss)

    dokument_thresh = cv.adaptiveThreshold(dokument_resize,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,9,6)
    # cv.imshow('Thresh',dokument_thresh)

    dokument_close = cv.morphologyEx(dokument_thresh, cv.MORPH_CLOSE, kernel)
    cv.imshow('Opening',dokument_close)

    cv.imwrite('dokumment_przetworzony.png',dokument_close)

if __name__ == '__main__':
    # cv.namedWindow("Filter Size")
    # cv.createTrackbar('Size', 'Filter Size',1,10,on_trackbar)
    # operacje_morfologiczne(black_white)
    # skanowanie_obrazu(brg_image)
    # filtr_Kuwahary(square_image,5)
    wdpo3()

    cv.waitKey(0)
    cv.destroyAllWindows()
