import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from camera_calibrate import load_camera_calibration
from perspective_transform import *
from thresholding import *
from moviepy.editor import VideoFileClip
from vehicle_detect_nn import VehicleDetector


def save_image(image, file_name):
    cv2.imwrite(file_name, image)


def bin_to_rgb(bin_image):
    return cv2.cvtColor(bin_image * 255, cv2.COLOR_GRAY2BGR)


def compose_images(dst_image, src_image, split_rows, split_columns, which_section):
    assert 0 < which_section <= split_rows * split_columns

    if split_rows > split_columns:
        newH = int(dst_image.shape[0] / split_rows)
        dim = (int(dst_image.shape[1] * newH / dst_image.shape[0]), newH)
    else:
        newW = int(dst_image.shape[1] / split_columns)
        dim = (newW, int(dst_image.shape[0] * newW / dst_image.shape[1]))

    if len(src_image.shape) == 2:
        srcN = bin_to_rgb(src_image)
    else:
        srcN = np.copy(src_image)

    img = cv2.resize(srcN, dim, interpolation=cv2.INTER_AREA)
    nr = (which_section - 1) // split_columns
    nc = (which_section - 1) % split_columns
    dst_image[nr * img.shape[0]:(nr + 1) * img.shape[0], nc * img.shape[1]:(nc + 1) * img.shape[1]] = img
    return dst_image


def plot_to_image(plt):
    plt.savefig('tmp_plt.png')
    img = cv2.imread('tmp_plt.png')
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


class LaneFinder(object):

    MAX_REUSED_FRAME = 5

    def __init__(self, save_original_images, object_detection_mask=lambda image: np.zeros_like(image),
                 camera_calibration_file="./output_images/camera_calibration_pickle.p"):
        camera_matrix, distortion = load_camera_calibration(camera_calibration_file)
        self.camera_matrix = camera_matrix
        self.distortion = distortion
        self.save_original_images = save_original_images
        self.process_counter = 0
        self.last_left_fits = []
        self.last_right_fits = []
        self.last_left_search_base = None
        self.last_right_search_base = None
        self.image_shape = None
        self.object_detection_mask = object_detection_mask

    def process_image(self, image):
        """
        :param image: Image with RGB color channels
        :return: new image with all lane line information
        """
        if self.image_shape is None:
            self.image_shape = image.shape

        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        final_image = np.zeros_like(image_bgr)
        compose_images(final_image, image_bgr, 2, 2, 1)

        if self.save_original_images:
            save_image(image_bgr, "./video_images/{}.jpg".format(self.process_counter))

        undistored_image = cv2.undistort(image_bgr, self.camera_matrix, self.distortion, None, self.camera_matrix)
        threshold_combined = pipeline(undistored_image)
        matrix, invent_matrix = calculate_transform_matrices(threshold_combined.shape[1], threshold_combined.shape[0])
        perspective_img = perspective_transform(threshold_combined, matrix)
        binary_warped = perspective_img
        left_fit, right_fit, left_search_base, right_search_base, histogram, out_img = \
            self.find_line_with_slide_window(binary_warped, self.last_left_search_base, self.last_right_search_base)
        self.last_left_search_base = left_search_base
        self.last_right_search_base = right_search_base

        left_fit, right_fit = self._compare_and_get_best_fit(left_fit, right_fit)
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        histogram_graph = self._add_history(out_img, histogram, ploty, left_fitx, right_fitx)
        compose_images(final_image, histogram_graph, 2, 2, 2)

        color_warp = self.apply_fit_to_road(binary_warped, ploty, left_fitx, right_fitx)
        left_curverad, right_curverad, center_distance = self._calculate_radius(ploty, left_fit, right_fit, left_fitx, right_fitx)

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = inversion_perspective_transform(color_warp, invent_matrix)
        # Combine the result with the original image
        result = cv2.addWeighted(undistored_image, 1, newwarp, 0.5, 0)
        compose_images(final_image, result, 2, 2, 3)

        object_detect_mask = np.zeros_like(undistored_image)
        object_detect_image = self.object_detection_mask(undistored_image)
        object_detect_image = cv2.addWeighted(object_detect_image, 1, object_detect_mask, 0.4, 0)

        compose_images(final_image, object_detect_image, 2, 2, 4)
        self._print_text(final_image, self.process_counter, left_curverad, right_curverad, center_distance)
        self.process_counter += 1

        return cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)

    @staticmethod
    def find_line_with_slide_window(binary_warped, leftx_base=None, rightx_base=None):
        # Assuming you have created a warped binary image called "binary_warped"
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        leftx_base, rightx_base = LaneFinder._line_search_base_position(
            histogram, leftx_base, rightx_base, peak_detect_offset=120)

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(binary_warped.shape[0] / nwindows)
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
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            rectangle_color = (0, 0, 250)
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), rectangle_color, 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), rectangle_color, 2)
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
                nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
                nonzerox < win_xright_high)).nonzero()[0]
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

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        return left_fit, right_fit, leftx_base, rightx_base, histogram, out_img

    @staticmethod
    def _line_search_base_position(histogram,
                                   last_know_leftx_base=None, last_know_rightx_base=None, peak_detect_offset=80):
        if last_know_leftx_base is None or last_know_rightx_base is None:
            midpoint = np.int(histogram.shape[0] // 2)
            leftx_base = np.argmax(histogram[:midpoint])
            rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        else:
            left_start, left_end, right_start, right_end = LaneFinder._range(
                len(histogram), last_know_leftx_base, last_know_rightx_base, peak_detect_offset)
            leftx_base = np.argmax(histogram[left_start:left_end]) + left_start
            rightx_base = np.argmax(histogram[right_start:right_end]) + right_start

        return leftx_base, rightx_base

    @staticmethod
    def _range(length, left, right, offset):
        left_start = left - offset
        if left_start < 0:
            left_start = 0
        left_end = left + offset + 1
        if left_end >= length:
            left_end = length - 1
        right_start = right - offset
        if right_start < 0:
            right_start = 0
        right_end = right + offset + 1
        if right_end >= length:
            right_end = length - 1

        return left_start, left_end, right_start, right_end

    def _calculate_radius(self, ploty, left_fit, right_fit, leftx, rightx):
        y_eval = np.max(ploty)
        left_curverad = ((1 + (2 * left_fit[0] * y_eval + left_fit[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit[0])
        right_curverad = ((1 + (2 * right_fit[0] * y_eval + right_fit[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit[0])

        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad_real = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * left_fit_cr[0])
        right_curverad_real = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * right_fit_cr[0])
        # Now our radius of curvature is in meters

        width = self.image_shape[1]
        height = self.image_shape[0]
        left = left_fit[0] * height ** 2 + left_fit[1] * height + left_fit[2]
        right = right_fit[0] * height ** 2 + right_fit[1] * height + right_fit[2]
        center_point = (left + right) / 2
        center_distance = (width / 2 - center_point) * xm_per_pix

        return left_curverad_real, right_curverad_real, center_distance

    def _print_text(self, final_image, image_index, left_curverad, right_curverad, center_distance):
        width = final_image.shape[1]
        height = final_image.shape[0]
        cv2.putText(final_image, "Image Index: {}".format(image_index),
                    (int(width / 2) + 40, int(height / 2) + 0 * 40 + 60), cv2.FONT_HERSHEY_SIMPLEX, 1, [255, 255, 255], 2)
        cv2.putText(final_image, "Center Distance:  {:6.2f} m".format(center_distance),
                    (int(width / 2) + 40, int(height / 2) + 1 * 40 + 60), cv2.FONT_HERSHEY_SIMPLEX, 1, [255, 255, 255],
                    2)
        cv2.putText(final_image, "Left Line Curve:  {:6.2f} m".format(left_curverad),
                    (int(width / 2) + 40, int(height / 2) + 2 * 40 + 60), cv2.FONT_HERSHEY_SIMPLEX, 1, [255, 255, 255],
                    2)
        cv2.putText(final_image, "Right Line Curve: {:6.2f} m".format(right_curverad),
                    (int(width / 2) + 40, int(height / 2) + 3 * 40 + 60), cv2.FONT_HERSHEY_SIMPLEX, 1, [255, 255, 255],
                    2)

    def _add_history(self, out_img, histogram, ploty, left_fitx, right_fitx):
        plt.figure()
        plt.imshow(cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB), cmap=None)
        plt.plot(720 - histogram, color='white', linewidth=3.0)
        plt.plot(left_fitx, ploty, color='yellow', linewidth=2.0)
        plt.plot(right_fitx, ploty, color='yellow', linewidth=2.0)
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.tight_layout(pad=0)
        return plot_to_image(plt)

    def _compare_and_get_best_fit(self, new_left_fit, new_right_fit):
        """
        LandFinder class will save last 5 "normal" fit, they new_left_fit and new_right_fit will
        compare with last normal fit, if they didn't off the center too much, we think the new fit
        is "normal" and system will allow to use it to following process, otherwise new fit will discard
        and last know "normal" fit will used for current frame.
        Only 5 frames maximum allocated to replace the new fit, if 5 frames still consider as "abnormal"
        this "abnormal" fit will still been used.
        :param new_left_fit:
        :param new_right_fit:
        :return:
        """
        if self._is_normal_fit(new_left_fit, new_right_fit):
            self.last_left_fits.append(new_left_fit)
            self.last_right_fits.append(new_right_fit)
            if len(self.last_left_fits) > self.MAX_REUSED_FRAME:
                self._drop_oldest_cached_fits()
            return new_left_fit, new_right_fit
        else:
            self._drop_oldest_cached_fits()
            left, right = self._recent_fits()
            print("fit abnormal, dropped. cached left{} cached right{} new left:{} new right:{}".format(
                left, right, new_left_fit, new_right_fit
            ))
            if left is None:
                print("Cache Fit exhausted, accepting new fit")
                return new_left_fit, new_right_fit
            return left, right

    def _drop_oldest_cached_fits(self):
        self.last_left_fits.pop(0)
        self.last_right_fits.pop(0)

    def _is_normal_fit(self, left_fit, right_fit):
        last_left_fit, last_right_fit = self._recent_fits()
        if last_left_fit is None:
            return True  # if nothing to compare, we think the new fit if normal
        return (last_left_fit[2] - 30 < left_fit[2] < last_left_fit[2] + 30) and \
               (last_right_fit[2] - 30 < right_fit[2] < last_right_fit[2] + 30)

    def _recent_fits(self):
        list_index = len(self.last_left_fits) - 1
        if list_index < 0:
            return None, None
        return self.last_left_fits[list_index], self.last_right_fits[list_index]

    def apply_fit_to_road(self, binary_warped, ploty, left_fitx, right_fitx):
        warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
        return color_warp


def remove_mp4_extension(file_name):
    return file_name.replace(".mp4", "")


if __name__ == "__main__":
    detector = VehicleDetector(img_rows=640, img_cols=960, weights_file="model_segn_small_0p72.h5")
    lane_finder = LaneFinder(save_original_images=False, object_detection_mask=detector.get_Unet_mask)
    video_file = 'project_video.mp4'
    # video_file = 'challenge_video.mp4'
    clip = VideoFileClip(video_file, audio=False)
    t_start = 0
    t_end = 0
    if t_end > 0.0:
        clip = clip.subclip(t_start=t_start, t_end=t_end)
    else:
        clip = clip.subclip(t_start=t_start)

    clip = clip.fl_image(lane_finder.process_image)
    clip.write_videofile("{}_output.mp4".format(remove_mp4_extension(video_file)), audio=False)

