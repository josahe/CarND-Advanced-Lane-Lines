# INTRODUCTION
The goal of this project is to write a software pipeline to identify the lane boundaries in a video from a front-facing camera on a car. 

### Steps
* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

# RELEVANT FILES
[Detailed writeup](https://github.com/josahe/CarND-Advanced-Lane-Lines/blob/master/advanced_lane_finding.md)
[Notebook](https://github.com/josahe/CarND-Advanced-Lane-Lines/blob/master/advanced_lane_finding.ipynb)

### Classes
* CameraCorrection (camera_correction.py)
* ImageTransforms (image_transforms.py)
* LineTracking (line_tracking.py)
