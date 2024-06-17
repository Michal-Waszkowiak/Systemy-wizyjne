import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('/home/michal/PycharmProjects/Systemy Wizyjne/LAB 01/lab01/detekcja_deskrypcja_dopasowanie/forward-5.bmp', cv.IMREAD_GRAYSCALE)
img_match = cv.imread('/home/michal/PycharmProjects/Systemy Wizyjne/LAB 01/lab01/detekcja_deskrypcja_dopasowanie/forward-1.bmp', cv.IMREAD_GRAYSCALE)

def TODO1_1():
    # Initiate FAST object with default values
    fast = cv.FastFeatureDetector_create()
    brief = cv.xfeatures2d.BriefDescriptorExtractor_create()

    # find and draw the keypoints
    kp = fast.detect(img, None)
    kp, des = brief.compute(img,kp)
    kp2 = fast.detect(img_match,None)
    kp2, des2 = brief.compute(img_match,kp2)
    img2 = cv.drawKeypoints(img, kp, None, color=(255, 0, 0))

    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    img_matched = cv.drawMatches(img, kp, img_match, kp2, matches[:20], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Print all default params
    print("Threshold: {}".format(fast.getThreshold()))
    print("nonmaxSuppression:{}".format(fast.getNonmaxSuppression()))
    print("neighborhood: {}".format(fast.getType()))
    print("Total Keypoints with nonmaxSuppression: {}".format(len(kp)))
    print(brief.descriptorSize())
    print(des.shape)

    # Disable nonmaxSuppression
    fast.setNonmaxSuppression(0)
    kp = fast.detect(img, None)

    print("Total Keypoints without nonmaxSuppression: {}".format(len(kp)))

    img3 = cv.drawKeypoints(img, kp, None, color=(255, 0, 0))

    while True:
        cv.imshow("Gray Base", img)
        cv.imshow("Fast Feature Detector", img2)
        cv.imshow("Fast Feature Detector - keypoints without nonmaxSuppression", img3)
        cv.imshow("Matched", img_matched)



        key_code = cv.waitKey(30)
        if key_code == 27:  # Escape
            break

    cv.destroyAllWindows()

def TODO1_2():
    # Initiate ORB detector
    orb = cv.ORB_create()

    # find the keypoints and descriptors with ORB
    kp, des = orb.detectAndCompute(img, None)
    kp2, des2 = orb.detectAndCompute(img_match, None)

    print(des)
    print(des2)

    # draw only keypoints location,not size and orientation
    img2 = cv.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)

    # create BFMatcher object
    bf = cv.BFMatcher(cv.NORM_HAMMING,crossCheck=True)

    # Match descriptors.
    matches = bf.match(des,des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw first 20 matches.
    img3 = cv.drawMatches(img, kp, img_match, kp2, matches[:20], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


    while True:
        cv.imshow("Gray Base", img)
        cv.imshow("ORB detector", img2)
        cv.imshow("Matches", img3)


        key_code = cv.waitKey(30)
        if key_code == 27:  # Escape
            break

    cv.destroyAllWindows()

def TODO1_3():
    img_base = img.copy()
    img_base_match = img_match.copy()
    sift = cv.SIFT_create()
    kp, des = sift.detectAndCompute(img,None)
    kp2, des2 = sift.detectAndCompute(img_base_match,None)
    print(des)

    bf = cv.BFMatcher()
    matches = bf.knnMatch(des,des2,k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.4 * n.distance:
            good.append([m])
    # cv.drawMatchesKnn expects list of lists as matches.
    img3 = cv.drawMatchesKnn(img_base, kp, img_base_match, kp2, good, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    img_base = cv.drawKeypoints(img, kp, img_base)


    while True:
        cv.imshow("Gray Base", img)
        cv.imshow("SIFT detector", img_base)
        cv.imshow("Matched", img3)


        key_code = cv.waitKey(30)
        if key_code == 27:  # Escape
            break

    cv.destroyAllWindows()



if __name__ == '__main__':
    # TODO1_1()
    # TODO1_2()
    TODO1_3()