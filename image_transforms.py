import numpy as np
import cv2
import helper_functions as hf

class ImageTransforms(object):
    """
    Calculate perspective transform coordinates using straight lane line images
    Perform perspective transformation of a camera image
    Perform binary threshold of a camera image
    """
    def __init__(self, calibration_images):
        self.image_shape = calibration_images[0].shape

        poly_coeffs = self.__calculate_best_fit(calibration_images)
        lft_coeffs=poly_coeffs[0]
        rht_coeffs=poly_coeffs[1]

        # Solving for x = (y-c)/m
        y_bl = int(self.image_shape[0])
        x_bl = int((y_bl-lft_coeffs[1])/lft_coeffs[0])
        y_tl = int(self.image_shape[0]-255)
        x_tl = int((y_tl-lft_coeffs[1])/lft_coeffs[0])
        y_tr = int(self.image_shape[0]-255)
        x_tr = int((y_tr-rht_coeffs[1])/rht_coeffs[0])
        y_br = int(self.image_shape[0])
        x_br = int((y_br-rht_coeffs[1])/rht_coeffs[0])

        self._y_bl = y_bl
        self._x_bl = x_bl
        self._y_tl = y_tl
        self._x_tl = x_tl
        self._y_tr = y_tr
        self._x_tr = x_tr
        self._y_br = y_br
        self._x_br = x_br

        # Source coordinates for perspective transforms
        self.src = np.float32([[x_bl, y_bl],
                               [x_tl, y_tl],
                               [x_tr, y_tr],
                               [x_br, y_br]])

        # Destination coordinates for perspective transforms
        self.dst = np.float32([[x_bl, self.image_shape[0]],
                               [x_bl, 0],
                               [x_br, 0],
                               [x_br, self.image_shape[0]]])

    def __calculate_best_fit(self, images):
        poly_coeffs=[]
        for image in images:
            poly_coeffs.append(self.pipeline(image, choose='poly_coeffs'))
        return np.mean(poly_coeffs, axis=0)

    def perspective_transform(self, image, inverse=False):
        """Perform perspective transform on image, given src
        and dst coordinates.
        """
        image_size = (image.shape[1], image.shape[0])

        if inverse is False:
            M = cv2.getPerspectiveTransform(self.src, self.dst)
        else:
            M = cv2.getPerspectiveTransform(self.dst, self.src)

        # keep same size as input image
        return cv2.warpPerspective(image, M, image_size, flags=cv2.INTER_NEAREST)

    def pipeline(self, image, choose='binary', threshold=''):
        height, width = image.shape[:2]

        # create binary threshold image
        binary = self.binary_threshold(image, choose=threshold)
        if choose == 'binary':
            return binary

        # mask unwanted regions
        vertices = np.array([[(0,height),(int(width/2),400),(width,height)]], dtype=np.int32)
        masked = hf.region_of_interest(binary, vertices)
        if choose == 'masked':
            return masked

        # calculate polynomail coefficients
        poly_coeffs = hf.calculate_poly_coeffs(masked)
        if choose == 'poly_coeffs':
            return poly_coeffs

        # draw lines and overlay on original image
        lines = hf.draw_lines(image, poly_coeffs)
        return hf.overlay_img(lines, image)

    def sobel(self, image, dim='x', ksize=5):
        if dim == 'x':
            sobel = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
        elif dim == 'y':
            sobel = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)
        abs_sobel = np.abs(sobel)
        return np.uint8(255*abs_sobel/np.max(abs_sobel))

    def apply_threshold(self, image, min_t, max_t):
        binary = np.zeros_like(image)
        binary[(image >= min_t) & (image <= max_t)] = 1
        return binary

    def binary_threshold(self, image, v_thresh=(200, 255), s_thresh=(170, 255),
                         x_thresh=(20, 255), choose=''):
        height, width = image.shape[:2]
        image = np.copy(image)

        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float)
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS).astype(np.float)

        v_channel = hsv[:,:,2]
        s_channel = hls[:,:,2]

        v_binary = self.apply_threshold(v_channel, v_thresh[0], v_thresh[1])
        s_binary = self.apply_threshold(s_channel, s_thresh[0], s_thresh[1])

        v_sobelx = self.sobel(v_channel)
        s_sobelx = self.sobel(s_channel)

        vx_binary = self.apply_threshold(v_sobelx, x_thresh[0], x_thresh[1])
        sx_binary = self.apply_threshold(s_sobelx, x_thresh[0], x_thresh[1])

        if choose == 'v':
            return v_channel
        elif choose == 'v_binary':
            return v_binary
        elif choose == 'vx_binary':
            return vx_binary
        elif choose == 's':
            return s_channel
        elif choose == 's_binary':
            return s_binary
        elif choose == 'sx_binary':
            return sx_binary

        binary = np.zeros_like(s_sobelx)
        binary[((s_binary==1) & (v_binary==1)) | ((sx_binary==1) & (vx_binary==1))] = 1
        return binary
