import numpy as np
import cv2
import helper_functions as hf

class CameraCorrection(object):
    """Wrapper class for OpenCV calibration and undistort functions.
    """

    def __init__(self):
        # Arrays to store object points and image points from all the images.
        self.objpoints = [] # 3d points in real world space
        self.imgpoints = [] # 2d points in image plane.

        self.image_shape = None
        self.mtx = None # calibration matrix
        self.dist = None # distortion coefficients

    def find_corners(self, image_files, draw_images=False):
        """Iterate through calibration images and search for chessboard corners.
        """
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((6*9,3), np.float32)
        objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

        for fname in image_files:
            img = cv2.imread(fname)
            gray = hf.grayscale(img)

            if self.image_shape is None:
                self.image_shape = gray.shape[::-1]

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

            # If found, add object points, image points
            if ret == True:
                self.objpoints.append(objp)
                self.imgpoints.append(corners)

                # Draw and display the corners
                if draw_images == True:
                    img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
                    cv2.imshow('img',img)
                    cv2.waitKey(250)

        cv2.destroyAllWindows()

    def calibrate_camera(self):
        """Calibrate camera using points in calibration images.
        """
        (ret, self.mtx, self.dist,
         rvecs, tvecs) = cv2.calibrateCamera(self.objpoints,
                                             self.imgpoints,
                                             self.image_shape,
                                             None, None)

    def undistort_image(self, image):
        """Undistort image using calibration matrix and
        distortion coefficients.
        """
        return cv2.undistort(image, self.mtx, self.dist, None, self.mtx)

    def undistort_images(self, images):
        """Undistort an array of images.
        """
        undistorted_images=[]
        for image in images:
            undistorted_images.append(self.undistort_image(image))
        return undistorted_images
