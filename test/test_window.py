import unittest
import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import time
from features import get_hog_features, extract_color_features, extract_hog_features, extract_features_all
from search_windows import search_windows, slide_window, draw_boxes
import matplotlib.image as mpimg
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


class SlideWindowSettings(object):
    def __init__(self, name, y_start_stop, size, rect_color, xy_overlap, thick):
        self.name = name
        self.y_start_stop = y_start_stop
        self.size = size
        self.rect_color = rect_color
        self.xy_overlap = xy_overlap
        self.thick = thick


def create_slide_window_settings(image_height, disable_overlap):
    def overlap(real_overlap):
        if disable_overlap:
            return (1., 1.)
        else:
            return real_overlap

    half = image_height // 2
    return [
        SlideWindowSettings("far", y_start_stop=(half, half + 130), size=(60, 60), rect_color=(0, 0, 200),
                            xy_overlap=overlap((0.5, 0.5)), thick=1),
        SlideWindowSettings("far", y_start_stop=(half, half + 180), size=(130, 80), rect_color=(0, 0, 200),
                            xy_overlap=overlap((0.5, 0.5)), thick=1),
        SlideWindowSettings("mid", y_start_stop=(half, half + 180), size=(170, 100), rect_color=(0, 0, 200),
                            xy_overlap=overlap((0.5, 0.5)), thick=1),
        SlideWindowSettings("mid left far away", y_start_stop=(half, half + 180), size=(170, 170), rect_color=(0, 0, 200),
                            xy_overlap=overlap((0.5, 0.5)), thick=1),
        SlideWindowSettings("mid", y_start_stop=(half, half + 180), size=(290, 140), rect_color=(0, 0, 200),
                            xy_overlap=overlap((0.5, 0.5)), thick=1),
        SlideWindowSettings("mid", y_start_stop=(half + 50, half + 300), size=(350, 200), rect_color=(0, 0, 200),
                            xy_overlap=(0.6, 0.6), thick=3)
    ]



class SearchWindowTest(unittest.TestCase):

    @staticmethod
    def apply_slide_window_to_file(file_name, slide_window_settings):
        image = mpimg.imread("../test_images/{}".format(file_name))
        image_height = image.shape[0]
        half = image_height // 2

        draw_image = np.copy(image)
        all_windows = []
        windows = slide_window(image, x_start_stop=[None, None], y_start_stop=slide_window_settings.y_start_stop,
                               xy_window=slide_window_settings.size, xy_overlap=slide_window_settings.xy_overlap)

        draw_image = draw_boxes(draw_image, windows,
                                color=slide_window_settings.rect_color, thick=slide_window_settings.thick)
        all_windows += windows

        plt.imshow(draw_image)
        plt.savefig("../output_images/slide_window/{}.png".format(file_name))


    def test_slide_window_close(self):
        SearchWindowTest.apply_slide_window_to_file(
            "746.jpg",
            create_slide_window_settings(720, False)[5]
        )

    def test_side_window_mid(self):
        SearchWindowTest.apply_slide_window_to_file(
            "test1.jpg",
            create_slide_window_settings(720, False)[4]
        )

    def test_search_window(self):
        image = mpimg.imread("../test_images/test1.jpg")
        height = image.shape[0]
        y_start_stop = (height // 2, height - 30)

        window_settings = [
            {"y_start_stop": (height // 2, height // 2 + 120), "size": (60, 60), "color": (0, 0, 200)},
            {"y_start_stop": (height // 2, height // 2 + 180), "size": (90, 90), "color": (0, 200, 200)},
            {"y_start_stop": (height // 2, height - 30), "size": (300, 160), "color": (0, 200, 0)},
            {"y_start_stop": (height // 2, height - 10), "size": (400, 180), "color": (200, 0, 0)},
        ]

        draw_image = np.copy(image)
        all_windows = []
        for settings in window_settings:
            windows = slide_window(image, x_start_stop=[None, None], y_start_stop=settings.get("y_start_stop"),
                               xy_window=settings.get("size"), xy_overlap=(0.5, 0.5))

            draw_image = draw_boxes(draw_image, windows, color=settings.get("color"), thick=3)
            all_windows += windows

        plt.imshow(draw_image)
        plt.savefig("../output_images/slide_window.png")
