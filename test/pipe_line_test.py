import unittest
import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from camera_calibrate import load_camera_calibration
from perspective_transform import *
from thresholding import *
from main import LaneFinder
from vehicle_detect_nn import VehicleDetector
import matplotlib.image as mpimg
from object_detect_yolo import YoloDetector


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


def show_image(image, cmap=None):
    plt.figure()
    plt.imshow(image, cmap)


def save_image(image, file_name):
    plt.figure()
    plt.imshow(image)
    plt.savefig(file_name)


def save_image_gray(image, file_name):
    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.savefig(file_name)


def plot_two_image(image1, image2):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()

    ax1.imshow(image1[0])
    ax1.set_title(image1[1], fontsize=40)

    ax2.imshow(image2[0])
    ax2.set_title(image2[1], fontsize=40)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

    return plt, ax1, ax2


class PipeLineTest(unittest.TestCase):
    def test_vehicle_detection(self):
        images = glob.glob('../test_images/*.jpg')
        for fname in images:
            detector = VehicleDetector(img_rows=640, img_cols=960, weights_file="../model_segn_small_0p72.h5")
            image = mpimg.imread(fname)
            result_pipe = image
            result_pipe = detector.get_BB_new_img(result_pipe)
            save_image(result_pipe, '../output_images/vehicle/{}.png'.format(os.path.basename(fname)))

    def test_perspective_transform(self):
        images = glob.glob('../test_images/*.jpg')
        for fname in images:
            img = cv2.imread(fname)
            matrix, invent_matrix = calculate_transform_matrices(img.shape[1], img.shape[0])
            perspective_img = perspective_transform(img, matrix)
            save_image(perspective_img, '../output_images/perspective/{}.png'.format(os.path.basename(fname)))

    def test_combine_with_or(self):
        a1 = np.array([[1, 0, 1]])
        a2 = np.array([[1, 1, 0]])
        a3 = combine_with_or(a1, a2)
        np.testing.assert_almost_equal([[1, 1, 1]], a3)

    def test_combine_with_and(self):
        a1 = np.array([[1, 0, 1]])
        a2 = np.array([[1, 1, 0]])
        result = np.array([[1, 0, 0]])
        a3 = combine_with_and(a1, a2)
        np.testing.assert_almost_equal([[1, 0, 0]], a3)

    def test_threshold(self):
        image = cv2.imread('../test_images/test1.jpg')
        # image = cv2.imread('../test_images/test1.jpg')
        hls_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS).astype(np.float)
        h_binary, l_binary, s_binary = hls_channel_threshold(
            hls_image, h_thresh=(170, 255), l_thresh=(190, 255),
            s_thresh=(170, 255))
        save_image_gray(h_binary, "../output_images/h_channel.png")
        save_image_gray(l_binary, "../output_images/l_channel.png")
        save_image_gray(s_binary, "../output_images/s_channel.png")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ksize = 25
        save_image_gray(mag_thresh(gray, ksize, (50, 255)), "../output_images/sobel_mag.png")
        save_image_gray(dir_threshold(gray, sobel_kernel=31, thresh=(0.7, 1.2)), "../output_images/sobel_dir.png")
        save_image_gray(combine_threshold(gray), "../output_images/sobel_combined.png")

        save_image_gray(
            abs_sobel_thresh(gray, "x", sobel_kernel=ksize, thresh=(50, 150)), "../output_images/sobel_x.png")
        save_image_gray(
            abs_sobel_thresh(gray, "y", sobel_kernel=ksize, thresh=(30, 100)), "../output_images/sobel_y.png")

        save_image_gray(
            combine_with_and(
                hls_channel_threshold(hls_image, l_thresh=(190, 255))[1],
                dir_threshold(gray, sobel_kernel=31, thresh=(0.7, 1.2))
            ), "../output_images/sobel_l+dir.png")
        save_image_gray(
            combine_with_and(
                hls_channel_threshold(hls_image, s_thresh=(170, 255))[2],
                abs_sobel_thresh(gray, orient='x', sobel_kernel=5, thresh=(10, 100))
            ), "../output_images/sobel_s+x.png")
        save_image_gray(
            combine_with_or(
                *bgr_channel_threshold(image)
            ), "../output_images/b+g+r_channels.png"
        )
        save_image_gray(
            pipeline(image), "../output_images/sobel_final_pipe_line.png")

    def test_threshold_pipe_line(self):
        images = glob.glob('../test_images/*.jpg')
        for fname in images:
            image = cv2.imread(fname)
            combined = pipeline(image)

            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
            f.tight_layout()

            ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            ax1.set_title('Original Image', fontsize=40)

            ax2.imshow(combined, cmap='gray')
            ax2.set_title('Pipeline Result', fontsize=40)
            plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
            plt.savefig("../output_images/thresholding/{}.png".format(os.path.basename(fname)))

    def test_lane_line(self):
        yolo = YoloDetector(model_path="../yolo/model_data/yolo.h5",
                            anchors_path="../yolo/model_data/yolo_anchors.txt",
                            classes_path="../yolo/model_data/coco_classes.txt",
                            font_file_name="../yolo/font/FiraMono-Medium.otf")
        images = glob.glob('../test_images/*.jpg')
        for fname in images:
            lane_finder = LaneFinder(save_original_images=False, object_detection_mask=yolo.process_image_array,
                                     camera_calibration_file="../output_images/camera_calibration_pickle.p")
            image = cv2.imread(fname)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            final_image = lane_finder.process_image(image)
            cv2.imwrite("../output_images/lane/combine_{}.png".format(os.path.basename(fname)),
                        cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))

    def test_find_line_one_image(self):
        fname = "../test_images/602.jpg"

        detector = VehicleDetector(img_rows=640, img_cols=960, weights_file="../model_segn_small_0p72.h5")

        lane_finder = LaneFinder(save_original_images=False, object_detection_mask=detector.get_Unet_mask,
                                 camera_calibration_file="../output_images/camera_calibration_pickle.p")
        image = cv2.imread(fname)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        final_image = lane_finder.process_image(image)
        cv2.imwrite("../output_images/lane/combine_{}.png".format(os.path.basename(fname)),
                    cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))


    def test_find_line_one_image_yolo(self):
        fname = "../test_images/602.jpg"

        yolo = YoloDetector(model_path="../yolo/model_data/yolo.h5",
                            anchors_path="../yolo/model_data/yolo_anchors.txt",
                            classes_path="../yolo/model_data/coco_classes.txt",
                            font_file_name="../yolo/font/FiraMono-Medium.otf")

        lane_finder = LaneFinder(save_original_images=False, object_detection_mask=yolo.process_image_array,
                                 camera_calibration_file="../output_images/camera_calibration_pickle.p")
        image = cv2.imread(fname)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        final_image = lane_finder.process_image(image)
        cv2.imwrite("../output_images/lane/combine_{}.png".format(os.path.basename(fname)),
                    cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))
        yolo.shutdown()

    def test_undistorted(self):
        camera_matrix, distortion = load_camera_calibration("../output_images/camera_calibration_pickle.p")

        images = glob.glob('../test_images/*.jpg')
        for fname in images:
            image = cv2.imread(fname)
            undistorted_image = cv2.undistort(image, camera_matrix, distortion, None, camera_matrix)
            plt, _, _ = plot_two_image(
                (cv2.cvtColor(image, cv2.COLOR_BGR2RGB), "original"),
                (cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2RGB), "undistorted"))
            plt.savefig("../output_images/undistort/{}.png".format(os.path.basename(fname)))

    def test_line_search_base_position_should_find_middle_point_if_no_last_knowledge(self):
        histogram = np.array([1, 2, 1, 3, 4, 3])
        left, right = LaneFinder._line_search_base_position(histogram, None, None)
        self.assertEqual(left, 1)
        self.assertEqual(right, 4)

    def test_line_search_base_position_should_find_peak_point_near_last_know_position(self):
        histogram = np.array([1, 4, 1, 2, 1, 3, 4, 3, 5, 3])
        left, right = LaneFinder._line_search_base_position(histogram, None, None)
        self.assertEqual(left, 1)
        self.assertEqual(right, 8)
        left, right = LaneFinder._line_search_base_position(
            histogram, last_know_leftx_base=4, last_know_rightx_base=6, peak_detect_offset=1)
        self.assertEqual(left, 5)
        self.assertEqual(right, 6)
        left, right = LaneFinder._line_search_base_position(
            histogram, last_know_leftx_base=4, last_know_rightx_base=9, peak_detect_offset=2)
        self.assertEqual(left, 6)
        self.assertEqual(right, 8)

