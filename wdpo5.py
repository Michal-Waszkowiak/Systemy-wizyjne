import cv2 as cv
import numpy as np

def empty_callback(value):
    pass

image_BRG = cv.imread('BRG.jpg', cv.IMREAD_GRAYSCALE)
image_canny = cv.imread('One_Ringjpg.jpg',cv.IMREAD_GRAYSCALE)
image_drone = cv.imread('drone_ship.jpg')

def prewitt_sobel(image):

    # Prewitt
    prewitt_matrix_x = np.array([[1,0,-1],
                                 [1,0,-1],
                                 [1,0,-1]])
    prewitt_matrix_y = np.array([[1, 1, 1],
                                 [0, 0, 0],
                                 [-1, -1, -1]])
    prewitt_x = cv.filter2D(image,cv.CV_32F,prewitt_matrix_x)
    prewitt_y = cv.filter2D(image,cv.CV_32F,prewitt_matrix_y)

    gradient_prewitt = np.sqrt(prewitt_x ** 2 + prewitt_y ** 2)
    gradient_dir_prewitt = np.arctan2(prewitt_y,prewitt_x)

    # Obliczenie pochodnych cząstkowych za pomocą maski Sobela
    sobel_kernel_x = np.array([[1, 0, -1],
                               [2, 0, -2],
                               [1, 0, -1]])
    sobel_kernel_y = np.array([[1, 2, 1],
                               [0, 0, 0],
                               [-1, -2, -1]])

    sobel_x = cv.filter2D(image, cv.CV_32F, sobel_kernel_x)
    sobel_y = cv.filter2D(image, cv.CV_32F, sobel_kernel_y)

    # Obliczenie gradientu i kierunku gradientu za pomocą maski Sobela
    gradient_sobel = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    gradient_direction_sobel = np.arctan2(sobel_y, sobel_x)

    gradient_prewitt_scaled = (gradient_prewitt / np.amax(gradient_prewitt) * 255).astype(np.uint8)
    gradient_sobel_scaled = (gradient_sobel / np.amax(gradient_sobel) * 255).astype(np.uint8)

    # Wyświetlenie wyników
    cv.imshow('Obraz wejściowy', image)
    cv.imshow('Prewitt - Gradient', gradient_prewitt_scaled)
    cv.imshow('Prewitt - Kierunek gradientu', gradient_dir_prewitt)
    cv.imshow('Sobel - Gradient', gradient_sobel_scaled)
    cv.imshow('Sobel - Kierunek gradientu', gradient_direction_sobel)


# cv.namedWindow('Base')
# cv.imshow('Base', image_drone)
# cv.createTrackbar('Low', 'Base', 0, 200, empty_callback)
# cv.createTrackbar('High', 'Base', 2, 100, empty_callback)

def canny(image):
    low_value = cv.getTrackbarPos('Low', 'Base')
    high_value = cv.getTrackbarPos('High', 'Base')

    edges = cv.Canny(image, low_value, high_value)

    cv.imshow('Canny', edges)

def Hough_line():
    image_hough = cv.imread('shapes.jpg')
    gray = cv.cvtColor(image_hough, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray,40,200,apertureSize=3)

    lines = cv.HoughLinesP(edges,1,np.pi/180,30,minLineLength=50,maxLineGap=10)
    for line in lines:
        x1,y1,x2,y2 = line[0]

        cv.line(image_hough, (x1, y1), (x2, y2), (0, 0, 255), 1)

    cv.imshow('Hough', image_hough)

def Hough_Circle():
    img = cv.imread('shapes.jpg', cv.IMREAD_GRAYSCALE)
    assert img is not None, "file could not be read, check with os.path.exists()"
    img = cv.medianBlur(img, 5)
    cimg = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

    circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, 20,
                              param1=100, param2=40, minRadius=0, maxRadius=0)

    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # draw the outer circle
        cv.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)

    cv.imshow('detected circles', cimg)

def wdpo5_1():
    image = cv.imread('drone_ship.jpg')
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    edges = cv.Canny(gray, 250, 255)
    contours, _ = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    cv.drawContours(image, contours, -1, (200, 120, 250), 1)
    cv.imshow('Ship Contours', image)

def wdpo5_2():
    image = cv.imread('fruit.jpg')
    image_gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    image_blur = cv.medianBlur(image_gray,9)
    ret, thresh = cv.threshold(image_blur,250,255,cv.THRESH_BINARY)
    kernel = np.ones((9,9),np.uint8)
    image_close = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel,iterations=2)
    contours, hierarchy = cv.findContours(image_close, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    for con in contours:
        area = cv.contourArea(con)

        if area < 100000:
            mask = np.zeros(image.shape[:2],dtype=np.uint8)
            cv.drawContours(mask,[con],-1,255,thickness=cv.FILLED)

            mean = cv.mean(image,mask)[:3]

            if (27 < mean[0] < 31) and (183 < mean[1] < 187) and (145 < mean[2] < 149):
                cv.drawContours(image, [con], -1, (0, 0, 255), 3)
            elif (22 < mean[0] < 26) and (123 < mean[1] < 127) and (233 < mean[2] < 237):
                cv.drawContours(image, [con], -1, (255, 0, 0), 3)


    # cv.imshow('Base', image)
    # cv.imshow('Gray',image_gray)
    # cv.imshow('Blur',image_blur)
    # cv.imshow('Thresh',thresh)
    # cv.imshow('Close',image_close)
    cv.imshow('Fruits', image)



def wdpo5_3():
    image = cv.imread('coins.jpg')
    image_gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    image_blur = cv.medianBlur(image_gray, 5)
    ret, thresh = cv.threshold(image_blur, 250, 255, cv.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    image_opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)
    image_color = cv.cvtColor(image_gray, cv.COLOR_GRAY2BGR)

    zl = 0
    gr = 0

    contours, _ = cv.findContours(image_opening, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    for contour in contours:
        area = cv.contourArea(contour)
        if 1 < area < 10000:
            gr = gr + 1
            cv.drawContours(image_color, [contour], -1, (0, 255, 0), 3)
        elif 10001 < area < 40000:
            zl = zl + 1
            cv.drawContours(image_color, [contour], -1, (0, 255, 0), 3)

    value = float(zl + 0.1 * gr)
    print('Kwota na obrazie: ' + f'{value:.2f}' + ' zł')
    cv.imshow('Detected money', image_color)

while True:

    # prewitt_sobel(image_BRG)
    # canny(image_drone)
    # Hough_line()
    # Hough_Circle()
    wdpo5_1()


    key_code = cv.waitKey(10)
    if key_code == 27:
        break

cv.destroyAllWindows()