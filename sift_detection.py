import cv2 as cv
import numpy as np


def find_fund():
    scene1 = cv.imread("scene_1.jpg")
    scene2 = cv.imread("scene_2.jpg")

    cv.namedWindow("Scene1", cv.WINDOW_NORMAL)
    cv.resizeWindow("Scene1", 700, 450)
    cv.imshow("Scene1", scene1)

    cv.namedWindow("Scene2", cv.WINDOW_NORMAL)
    cv.resizeWindow("Scene2", 700, 450)
    cv.imshow("Scene2", scene2)
    cv.waitKey(0)
    cv.destroyAllWindows()

    sift = cv.xfeatures2d.SIFT_create()

    scene1_kp, object_descriptors = sift.detectAndCompute(scene1, None)
    scene2_kp, scene_descriptors = sift.detectAndCompute(scene2, None)

    scene1_keypoints_drawed = scene1.copy()
    scene2_keypoints_drawed = scene2.copy()

    cv.drawKeypoints(scene1, scene1_kp, scene1_keypoints_drawed)
    cv.drawKeypoints(scene2, scene2_kp, scene2_keypoints_drawed)

    keypoints_image = np.concatenate((scene1_keypoints_drawed, scene2_keypoints_drawed), axis=1)

    cv.namedWindow("keypoints", cv.WINDOW_NORMAL)
    cv.resizeWindow("keypoints", 1700, 550)
    cv.imshow("keypoints", keypoints_image)
    cv.waitKey()
    cv.destroyAllWindows()

    descriptor_matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
    knn_matches = descriptor_matcher.knnMatch(object_descriptors, scene_descriptors, 2)

    matches = []
    pts1 = []
    pts2 = []
    for x, y in knn_matches:
        if x.distance < 0.75 * y.distance:
            matches.append(x)
            pts2.append(scene2_kp[x.trainIdx].pt)
            pts1.append(scene1_kp[x.queryIdx].pt)

    pts1 = np.float32(pts1)
    pts2 = np.float32(pts2)
    F, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_RANSAC)

    matches_img = np.empty((max(scene1.shape[0], scene2.shape[0]), scene1.shape[1] + scene2.shape[1], 3),
                           dtype=np.uint8)
    cv.drawMatches(scene1, scene1_kp, scene2, scene2_kp, matches, matches_img,
                   flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    cv.namedWindow("good matches", cv.WINDOW_NORMAL)
    cv.resizeWindow("good matches", 1700, 550)
    cv.imshow("good matches", matches_img)
    cv.waitKey()
    cv.destroyAllWindows()

    return F, pts1, pts2






