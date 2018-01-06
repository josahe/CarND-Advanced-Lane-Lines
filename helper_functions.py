import numpy as np
import cv2
from matplotlib import pyplot as plt

def visualise(images, titles, nrows, ncols, gray=False):
    f, axarr = plt.subplots(nrows, ncols)#, figsize=(20,10))
    if not (nrows == ncols == 1):
        for i in range(0, nrows, ncols):
            for j in range(0, ncols):
                idx = i+j
                if gray is False:
                    axarr[i, j].imshow(images[idx])
                else:
                    axarr[i, j].imshow(images[idx], cmap='gray')
                axarr[i, j].set_title(titles[idx])
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    else:
        axarr.imshow(images)
    plt.show()

def grayscale(img, is_cv2=True):
    if is_cv2 is True:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    return cv2.bitwise_and(img, mask)

def draw_lines(img, coeffs):
    lft_poly = np.poly1d(coeffs[0])
    rht_poly = np.poly1d(coeffs[1])
    line_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    imshape = img.shape
    cv2.line(line_image,
             (0, int(lft_poly(0))),
             (int(imshape[1]/2), int(lft_poly(imshape[1]/2))),
             [255, 0, 0], 3)
    cv2.line(line_image,
             (imshape[1], int(rht_poly(imshape[1]))),
             (int(imshape[1]/2), int(rht_poly(imshape[1]/2))),
             [255, 0, 0], 3)
    return line_image

def overlay_img(img, initial_img, α=0.8, β=1., λ=0.):
    return cv2.addWeighted(initial_img, α, img, β, λ)

def calculate_poly_coeffs(image):
    rho = 1
    theta = np.pi/180
    threshold = 15
    min_line_len = 5
    max_line_gap = 5

    lft = {'x1':[], 'x2':[], 'y1':[], 'y2':[]}
    rht = {'x1':[], 'x2':[], 'y1':[], 'y2':[]}

    lines = cv2.HoughLinesP(image, rho, theta, threshold, np.array([]),
                            minLineLength=min_line_len, maxLineGap=max_line_gap)

    for line in lines:
        for x1,y1,x2,y2 in line:
            slope = ((y2-y1)/(x2-x1))
            if(slope < 0):
                lft['x1'].append(x1)
                lft['x2'].append(x2)
                lft['y1'].append(y1)
                lft['y2'].append(y2)
            elif(slope > 0):
                rht['x1'].append(x1)
                rht['x2'].append(x2)
                rht['y1'].append(y1)
                rht['y2'].append(y2)

    lftx = np.concatenate([lft['x1'], lft['x2']])
    lfty = np.concatenate([lft['y1'], lft['y2']])
    rhtx = np.concatenate([rht['x1'], rht['x2']])
    rhty = np.concatenate([rht['y1'], rht['y2']])

    return np.polyfit(lftx, lfty, 1), np.polyfit(rhtx, rhty, 1)


#image = ...
#height, width = image.shape[:2]
#left_image = image[0:height, 0:width/2+50]
#right_image = image[0:height, width/2-50:width]
#stitched = np.concatenate((left_image[:, :width/2], right_image[:, 50:]), axis=1)
#if not np.array_equal(image, stitched):
#    raise ArithmeticError("Array's are not equal")
