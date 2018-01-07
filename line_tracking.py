import numpy as np
import cv2
from collections import deque

class LineTracking(object):
    def __init__(self, image_transforms, smoothing_buffer_size=1):
        # Calibrated ImageTransform object
        self.image_transforms = image_transforms
        # The size of the frame buffer for smoothing (averaging) of measurements
        self.smoothing_buffer_size = smoothing_buffer_size

        # x values averaged over the last n iterations
        self.left_best_x = None
        self.right_best_x = None

        # polynomial coefficients averaged over the last n iterations
        self.left_best_fit = None
        self.right_best_fit = None

        # polynomial coefficients of last n iterations
        self.left_fit_buffer = deque(self.smoothing_buffer_size*[[np.nan, np.nan, np.nan]],
                                     self.smoothing_buffer_size)
        self.right_fit_buffer = deque(self.smoothing_buffer_size*[[np.nan, np.nan, np.nan]],
                                      self.smoothing_buffer_size)

        # car position relative to lane
        self.line_base_pos = None
        self.lane_midpoint = None
        self.image_centre = self.image_transforms.image_shape[1]/2

        # Define conversions in x and y from pixels space to meters
        self.ym_per_pix = 20/720 # meters per pixel in y dimension
        self.xm_per_pix = 3.7/896 # meters per pixel in x dimension

        # was the line detected in the last iteration?
        self.detected = False
        self.counter = 0
        self.nframe = 0
        self.lost_lane_count = 0

        # radius of curvature of the line in some units
        self.left_curverad = None
        self.right_curverad = None
        self.radius_of_curvature = None

        ## Left lane

        # x values of the last n fits of the line
        self.left_recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.left_bestx = None
        # polynomial coefficients for the most recent fit
        self.left_current_fit = [np.array([False])]
        # difference in fit coefficients between last and new fits
        self.left_diffs = np.array([0,0,0], dtype='float')
        # x values for detected line pixels
        self.left_allx = None
        # y values for detected line pixels
        self.left_ally = None

        ## Right lane

        # x values of the last n fits of the line
        self.right_recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.right_bestx = None
        # polynomial coefficients for the most recent fit
        self.right_current_fit = [np.array([False])]
        # difference in fit coefficients between last and new fits
        self.right_diffs = np.array([0,0,0], dtype='float')
        # x values for detected line pixels
        self.right_allx = None
        # y values for detected line pixels
        self.right_ally = None

    def sliding_window(self, binary_warped):
        if self.detected is not True:
            self.sliding_window_start(binary_warped)
            self.detected = True
        else:
            self.sliding_window_optimised(binary_warped)

    def sliding_window_simple(self, binary_warped):
        self.sliding_window_start(binary_warped)

    def sliding_window_start(self, binary_warped):
        """Assuming image is binary warped.
        """
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(binary_warped.shape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
            (0,255,0), 2)
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
            (0,255,0), 2)
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        try:
            # Fit a second order polynomial to each lane line
            left_current_fit = np.polyfit(lefty, leftx, 2)
            right_current_fit = np.polyfit(righty, rightx, 2)
            self.lost_lane_count = 0
        except TypeError:
            self.lost_lane_count += 1
            if self.lost_lane_count > self.smoothing_buffer_size:
                raise ValueError('Could not find lane lines for '+
                                  str(self.smoothing_buffer_size)+
                                  ' consecutive frames')
            left_current_fit = [np.nan, np.nan, np.nan]
            right_current_fit = [np.nan, np.nan, np.nan]
            self.detected = False

        if self.sanity_check_radius(left_current_fit, right_current_fit, binary_warped) is not True:
            left_current_fit = [np.nan, np.nan, np.nan]
            right_current_fit = [np.nan, np.nan, np.nan]
            self.detected = False

        self.left_fit_buffer.append(left_current_fit)
        self.right_fit_buffer.append(right_current_fit)

        self.left_best_fit = np.nanmean(self.left_fit_buffer, axis=0)
        self.right_best_fit = np.nanmean(self.right_fit_buffer, axis=0)

         # Generate x and y values for plotting
        self.ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
        self.left_best_x = self.left_best_fit[0]*self.ploty**2 + self.left_best_fit[1]*self.ploty + self.left_best_fit[2]
        self.right_best_x = self.right_best_fit[0]*self.ploty**2 + self.right_best_fit[1]*self.ploty + self.right_best_fit[2]

        self.lane_midpoint = self.left_best_x[-1] + (self.right_best_x[-1] - self.left_best_x[-1]) / 2

    def sliding_window_optimised(self, binary_warped):
        """Assuming image is binary warped and sliding_window method
        was used on previous frame.
        """
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        left_lane_inds = ((nonzerox > (self.left_best_fit[0]*(nonzeroy**2) + self.left_best_fit[1]*nonzeroy +
            self.left_best_fit[2] - margin)) & (nonzerox < (self.left_best_fit[0]*(nonzeroy**2) +
            self.left_best_fit[1]*nonzeroy + self.left_best_fit[2] + margin)))

        right_lane_inds = ((nonzerox > (self.right_best_fit[0]*(nonzeroy**2) + self.right_best_fit[1]*nonzeroy +
            self.right_best_fit[2] - margin)) & (nonzerox < (self.right_best_fit[0]*(nonzeroy**2) +
            self.right_best_fit[1]*nonzeroy + self.right_best_fit[2] + margin)))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        try:
            # Fit a second order polynomial to each lane line
            left_current_fit = np.polyfit(lefty, leftx, 2)
            right_current_fit = np.polyfit(righty, rightx, 2)
            self.lost_lane_count = 0
        except TypeError:
            self.lost_lane_count += 1
            if self.lost_lane_count > self.smoothing_buffer_size:
                raise ValueError('Could not find lane lines for '+
                                  str(self.smoothing_buffer_size)+
                                  ' consecutive frames')
            left_current_fit = [np.nan, np.nan, np.nan]
            right_current_fit = [np.nan, np.nan, np.nan]
            self.detected = False

        if self.sanity_check_radius(left_current_fit, right_current_fit, binary_warped) is not True:
            left_current_fit = [np.nan, np.nan, np.nan]
            right_current_fit = [np.nan, np.nan, np.nan]
            self.detected = False

        self.left_fit_buffer.append(left_current_fit)
        self.right_fit_buffer.append(right_current_fit)

        self.left_best_fit = np.nanmean(self.left_fit_buffer, axis=0)
        self.right_best_fit = np.nanmean(self.right_fit_buffer, axis=0)

        # Generate x and y values for plotting
        self.ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
        self.left_best_x = self.left_best_fit[0]*self.ploty**2 + self.left_best_fit[1]*self.ploty + self.left_best_fit[2]
        self.right_best_x = self.right_best_fit[0]*self.ploty**2 + self.right_best_fit[1]*self.ploty + self.right_best_fit[2]

        self.lane_midpoint = self.left_best_x[-1] + (self.right_best_x[-1] - self.left_best_x[-1]) / 2

    def sanity_check_radius(self, left_current_fit, right_current_fit, binary_warped):
        threshold = 500

        check_ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
        check_left_x = left_current_fit[0]*check_ploty**2 + left_current_fit[1]*check_ploty + left_current_fit[2]
        check_right_x = right_current_fit[0]*check_ploty**2 + right_current_fit[1]*check_ploty + right_current_fit[2]

        if(self.left_best_x is not None and np.abs(np.mean(check_left_x) - np.mean(self.left_best_x)) > threshold or
           self.right_best_x is not None and np.abs(np.mean(check_right_x) - np.mean(self.right_best_x)) > threshold):
            return False
        return True

    def sanity_check_offset(self, left_current_fit, right_current_fit, binary_warped):
        threshold = 500

        check_ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
        check_left_x = left_current_fit[0]*check_ploty**2 + left_current_fit[1]*check_ploty + left_current_fit[2]
        check_right_x = right_current_fit[0]*check_ploty**2 + right_current_fit[1]*check_ploty + right_current_fit[2]

        check_lane_midpoint = check_left_x[-1] + (check_right_x[-1] - check_left_x[-1]) / 2

        if self.lane_midpoint is not None and np.abs(check_lane_midpoint - self.lane_midpoint) > threshold:
            return False
        return True

    def measure_curvature(self, true_scale=True):
        """Determine the curvature of the lane and vehicle position with respect to center.
        """
        # Define y-value where we want radius of curvature
        # I'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(self.ploty)
        left_curverad = ((1 + (2*self.left_best_x[0]*y_eval + self.left_best_x[1])**2)**1.5) / np.absolute(2*self.left_best_x[0])
        right_curverad = ((1 + (2*self.right_best_x[0]*y_eval + self.right_best_x[1])**2)**1.5) / np.absolute(2*self.right_best_x[0])
        if true_scale is False:
            self.left_curverad = left_curverad
            self.right_curverad = right_curverad
            self.radius_of_curvature = (left_curverad + right_curverad) / 2
            self.line_base_pos = (self.image_centre - self.lane_midpoint)
            return

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(self.ploty*self.ym_per_pix, self.left_best_x*self.xm_per_pix, 2)
        right_fit_cr = np.polyfit(self.ploty*self.ym_per_pix, self.right_best_x*self.xm_per_pix, 2)
        # Calculate the new radii of curvature
        self.left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*self.ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        self.right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*self.ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

        self.radius_of_curvature = (self.left_curverad + self.right_curverad)/2
        self.line_base_pos = (self.image_centre - self.lane_midpoint)*self.xm_per_pix

    def warp_lanes_back(self, image):
        """Warp the detected lane boundaries back onto the original image.
        """
        # Create an image to draw the lines on
        binary_image = self.image_transforms.pipeline(image)
        warp_zero = np.zeros_like(binary_image).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([self.left_best_x, self.ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([self.right_best_x, self.ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = self.image_transforms.perspective_transform(color_warp, inverse=True)

        # Combine the result with the original image
        result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)

        return result

    def text(self, image, text, ypos):
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10,ypos)
        fontScale              = 2
        fontColor              = (255,255,255)
        lineType               = 8

        return cv2.putText(image, text, bottomLeftCornerOfText, font,
                           fontScale, fontColor, lineType)
