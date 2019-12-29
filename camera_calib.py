import numpy as np
import cv2 as cv
import glob


def camera_calibration(cam_no):

    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6 * 8, 3), np.float32)
    objp[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2)
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    key = cv.waitKey(1)
    webcam = cv.VideoCapture(cam_no)
    img_counter = 0
    print("\nTake 20 calibration images!\nPress 's' to take an image or press 'q' to use saved images")
    while True:
        check, frame = webcam.read()
        cv.imshow("Capturing", frame)
        key = cv.waitKey(1)
        if key == ord('s'):
            img_counter += 1
            cv.imwrite('./Calibration_images/calibration_img_'+str(img_counter)+'.jpg', frame)
            print("Image saved!")
            if img_counter == 20:
                webcam.release()
                break
        elif key == ord('q'):
            webcam.release()
            break
    cv.destroyAllWindows()
    img_counter = 0

    images = glob.glob('./Calibration_images/*.jpg')
    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        img_counter += 1

        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (8, 6), None)
        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners)

            # Draw and display the corners
            cv.drawChessboardCorners(img, (8, 6), corners2, ret)
            cv.imshow('Calibration image' + str(img_counter), img)
            cv.waitKey(500)
    cv.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    img = cv.imread('./Calibration_images/calibration_img_2.jpg')
    h, w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    dst = cv.undistort(img, mtx, dist, None, newcameramtx)

    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    cv.imwrite('./Calibration_images/calibresult.png', dst)
    cv.imshow('Undistorted image 1', dst)
    cv.waitKey(1650)

    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        mean_error += error


    return mtx, dist