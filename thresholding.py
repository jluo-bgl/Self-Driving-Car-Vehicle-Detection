import cv2
import numpy as np


# Define a function that takes an image, gradient orientation,
# and threshold min / max values.
def abs_sobel_thresh(gray, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # Return the result
    return binary_output


# Define a function to return the magnitude of the gradient
# for a given sobel kernel size and threshold values
def mag_thresh(gray, sobel_kernel=3, mag_thresh=(0, 255)):
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag) / 255
    gradmag = (gradmag / scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output


# Define a function to threshold an image for a given range and Sobel kernel
def dir_threshold(gray, sobel_kernel=3, thresh=(0, np.pi / 2)):
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output


def combine_threshold(gray_image):
    # Choose a Sobel kernel size
    ksize = 5  # Choose a larger odd number to smooth gradient measurements

    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(gray_image, orient='x', sobel_kernel=ksize, thresh=(10, 100))
    grady = abs_sobel_thresh(gray_image, orient='y', sobel_kernel=ksize, thresh=(20, 100))
    mag_binary = mag_thresh(gray_image, sobel_kernel=ksize, mag_thresh=(30, 100))
    dir_binary = dir_threshold(gray_image, sobel_kernel=ksize, thresh=(0.7, 1.3))

    combined = np.zeros_like(dir_binary)

    combined[((gradx == 1) & (grady == 1))
             | ((gradx == 1) & (dir_binary == 1))
             | ((mag_binary == 1) & (dir_binary == 1))] = 1

    return combined


def hls_channel_threshold(hls_img, h_thresh=(170, 255), l_thresh=(170, 255), s_thresh=(170, 255)):
    hls_img = np.copy(hls_img)
    h_channel = hls_img[:, :, 0]
    l_channel = hls_img[:, :, 1]
    s_channel = hls_img[:, :, 2]

    h_binary = np.zeros_like(h_channel)
    h_binary[(h_channel >= h_thresh[0]) & (h_channel <= h_thresh[1])] = 1

    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= l_thresh[0]) & (l_channel <= l_thresh[1])] = 1

    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    return h_binary, l_binary, s_binary


def bgr_channel_threshold(bgr_img, b_thresh=(250, 255), g_thresh=(250, 255), r_thresh=(250, 255)):
    bgr_img = np.copy(bgr_img)
    b_channel = bgr_img[:, :, 0]
    g_channel = bgr_img[:, :, 1]
    r_channel = bgr_img[:, :, 2]

    b_binary = np.zeros_like(b_channel)
    b_binary[(b_channel >= b_thresh[0]) & (b_channel <= b_thresh[1])] = 1

    g_binary = np.zeros_like(g_channel)
    g_binary[(g_channel >= g_thresh[0]) & (g_channel <= g_thresh[1])] = 1

    r_binary = np.zeros_like(r_channel)
    r_binary[(r_channel >= r_thresh[0]) & (r_channel <= r_thresh[1])] = 1
    return b_binary, g_binary, r_binary


def combine_with_or(*channels):
    combined = None
    for channel in channels:
        if combined is None:
            combined = np.copy(channel)
        else:
            combined[(combined == 1) | (channel == 1)] = 1

    return combined


def combine_with_and(*channels):
    combined = None
    for channel in channels:
        if combined is None:
            combined = np.copy(channel)
        else:
            new = np.zeros_like(combined)
            new[(combined == 1) & (channel == 1)] = 1
            combined = new

    return combined


def pipeline(img):
    img = np.copy(img)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hls_image = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float)

    combined = combine_with_or(
        abs_sobel_thresh(gray_image, orient='x', sobel_kernel=25, thresh=(50, 150)),
        combine_with_or(
            *bgr_channel_threshold(img, b_thresh=(220, 255), g_thresh=(220, 255), r_thresh=(220, 255))
        ),
        combine_with_and(
            hls_channel_threshold(hls_image, s_thresh=(170, 255))[2],
            abs_sobel_thresh(gray_image, orient='x', sobel_kernel=5, thresh=(10, 100))
        )
    )
    return combined
