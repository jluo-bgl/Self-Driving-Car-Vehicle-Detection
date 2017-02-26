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


class FeaturesTest(unittest.TestCase):

    def test_hog_rect(self):
        gray_image = np.array([
            [255, 255, 255, 255, 255, 255],
            [255, 0,   0,   255, 255, 255],
            [255, 0,   0,   255, 255, 255],
            [255, 255, 255, 255, 255, 255]
        ])
        features, hog_image = get_hog_features(gray_image, orient=9,
                                               pix_per_cell=2, cell_per_block=2,
                                               vis=True, feature_vec=False)
        self.assertTupleEqual(features.shape, (1, 2, 2, 2, 9))
        plt.figure()
        plt.subplot(121)
        plt.imshow(gray_image, cmap='gray')
        plt.title('Example Car Image')
        plt.subplot(122)
        plt.imshow(hog_image, cmap='gray')
        plt.title('HOG Visualization')
        plt.show()

    def test_hog_with_car(self):
        image = mpimg.imread("../test_images/car1.png")
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        features, hog_image = get_hog_features(gray_image, orient=9,
                                               pix_per_cell=8, cell_per_block=2,
                                               vis=True, feature_vec=False)
        fig = plt.figure()
        plt.subplot(121)
        plt.imshow(gray_image, cmap='gray')
        plt.title('Example Car Image')
        plt.subplot(122)
        plt.imshow(hog_image, cmap='gray')
        plt.title('HOG Visualization')
        plt.savefig("../output_images/car1_hog.png")
        self.assertTupleEqual(gray_image.shape, (64, 64))
        self.assertTupleEqual(features.shape, (7, 7, 2, 2, 9))

    def test_feature_normalize_should_have_similar_max_min_range(self):
        car_file_names = glob.glob('../data/vehicles/*/*.png')
        notcar_file_names = glob.glob('../data/non-vehicles/*/*.png')

        car_features = extract_color_features(car_file_names, cspace='RGB')
        notcar_features = extract_color_features(notcar_file_names, cspace='RGB')

        X = np.vstack((car_features, notcar_features)).astype(np.float64)
        # Fit a per-column scaler

        X_scaler = StandardScaler().fit(X)
        # Apply the scaler to X

        scaled_X = X_scaler.transform(X)
        car_ind = np.random.randint(0, len(car_file_names))
        # Plot an example of raw and scaled features
        fig = plt.figure(figsize=(12, 4))
        plt.subplot(131)
        plt.imshow(mpimg.imread(car_file_names[car_ind]))
        plt.title('Original Image')
        plt.subplot(132)
        plt.plot(X[car_ind])
        plt.title('Raw Features')
        plt.subplot(133)
        plt.plot(scaled_X[car_ind])
        plt.title('Normalized Features')
        fig.tight_layout()
        plt.savefig("../output_images/feature_normalize.png")


    def test_svc_predict_should_have_acturacy_91(self):
        car_file_names = glob.glob('../data/vehicles/*/*.png')
        notcar_file_names = glob.glob('../data/non-vehicles/*/*.png')

        spatial = 32
        histbin = 32
        car_features = extract_color_features(car_file_names, cspace='RGB', spatial_size=(spatial, spatial),
                                              hist_bins=histbin)
        notcar_features = extract_color_features(notcar_file_names, cspace='RGB', spatial_size=(spatial, spatial),
                                                 hist_bins=histbin)
        X = np.vstack((car_features, notcar_features)).astype(np.float64)
        # Fit a per-column scaler

        X_scaler = StandardScaler().fit(X)
        # Apply the scaler to X

        scaled_X = X_scaler.transform(X)

        # Define the labels vector
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


        # Split up data into randomized training and test sets
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(
            scaled_X, y, test_size=0.2, random_state=rand_state)

        print('Using spatial binning of:', spatial,
              'and', histbin, 'histogram bins')
        print('Feature vector length:', len(X_train[0]))
        # Use a linear SVC
        svc = LinearSVC()
        # Check the training time for the SVC
        t = time.time()
        svc.fit(X_train, y_train)
        t2 = time.time()
        print(round(t2 - t, 2), 'Seconds to train SVC...')
        # Check the score of the SVC
        print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
        # Check the prediction time for a single sample
        t = time.time()
        n_predict = 10
        print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
        print('For these', n_predict, 'labels: ', y_test[0:n_predict])
        t2 = time.time()
        print(round(t2 - t, 5), 'Seconds to predict', n_predict, 'labels with SVC')


    def test_hog_should_give_98_acturacy(self):
        car_file_names = glob.glob('../data/vehicles/*/*.png')
        notcar_file_names = glob.glob('../data/non-vehicles/*/*.png')

        colorspace = 'HSV'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb

        orient = 9
        pix_per_cell = 8
        cell_per_block = 2
        hog_channel = "ALL"  # Can be 0, 1, 2, or "ALL"

        t = time.time()
        car_features = extract_hog_features(car_file_names, cspace=colorspace, orient=orient,
                                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                        hog_channel=hog_channel)
        notcar_features = extract_hog_features(notcar_file_names, cspace=colorspace, orient=orient,
                                           pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                           hog_channel=hog_channel)
        t2 = time.time()
        print(round(t2 - t, 2), 'Seconds to extract HOG features...')
        # Create an array stack of feature vectors
        X = np.vstack((car_features, notcar_features)).astype(np.float64)
        # Fit a per-column scaler
        X_scaler = StandardScaler().fit(X)
        # Apply the scaler to X
        scaled_X = X_scaler.transform(X)

        # Define the labels vector
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

        # Split up data into randomized training and test sets
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(
            scaled_X, y, test_size=0.2, random_state=rand_state)

        print('Using:', orient, 'orientations', pix_per_cell,
              'pixels per cell and', cell_per_block, 'cells per block')
        print('Feature vector length:', len(X_train[0]))
        # Use a linear SVC
        svc = LinearSVC()
        # Check the training time for the SVC
        t = time.time()
        svc.fit(X_train, y_train)
        t2 = time.time()
        print(round(t2 - t, 2), 'Seconds to train SVC...')
        # Check the score of the SVC
        print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
        # Check the prediction time for a single sample
        t = time.time()
        n_predict = 10
        print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
        print('For these', n_predict, 'labels: ', y_test[0:n_predict])
        t2 = time.time()
        print(round(t2 - t, 5), 'Seconds to predict', n_predict, 'labels with SVC')


    def test_search_features(self):
        ### TODO: Tweak these parameters and see how the results change.
        car_file_names = glob.glob('../data/vehicles/*/*.png')
        notcar_file_names = glob.glob('../data/non-vehicles/*/*.png')

        color_space = 'HSV'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        orient = 9  # HOG orientations
        pix_per_cell = 8  # HOG pixels per cell
        cell_per_block = 2  # HOG cells per block
        hog_channel = "ALL"  # Can be 0, 1, 2, or "ALL"
        spatial_size = (16, 16)  # Spatial binning dimensions
        hist_bins = 16  # Number of histogram bins
        spatial_feat = True  # Spatial features on or off
        hist_feat = True  # Histogram features on or off
        hog_feat = True  # HOG features on or off
        y_start_stop = [None, None]  # Min and max in y to search in slide_window()

        car_features = extract_features_all(car_file_names, color_space=color_space,
                                        spatial_size=spatial_size, hist_bins=hist_bins,
                                        orient=orient, pix_per_cell=pix_per_cell,
                                        cell_per_block=cell_per_block,
                                        hog_channel=hog_channel, spatial_feat=spatial_feat,
                                        hist_feat=hist_feat, hog_feat=hog_feat)
        notcar_features = extract_features_all(notcar_file_names, color_space=color_space,
                                           spatial_size=spatial_size, hist_bins=hist_bins,
                                           orient=orient, pix_per_cell=pix_per_cell,
                                           cell_per_block=cell_per_block,
                                           hog_channel=hog_channel, spatial_feat=spatial_feat,
                                           hist_feat=hist_feat, hog_feat=hog_feat)

        X = np.vstack((car_features, notcar_features)).astype(np.float64)
        # Fit a per-column scaler
        X_scaler = StandardScaler().fit(X)
        # Apply the scaler to X
        scaled_X = X_scaler.transform(X)

        # Define the labels vector
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

        # Split up data into randomized training and test sets
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(
            scaled_X, y, test_size=0.2, random_state=rand_state)

        print('Using:', orient, 'orientations', pix_per_cell,
              'pixels per cell and', cell_per_block, 'cells per block')
        print('Feature vector length:', len(X_train[0]))
        # Use a linear SVC
        svc = LinearSVC()
        # Check the training time for the SVC
        t = time.time()
        svc.fit(X_train, y_train)
        t2 = time.time()
        print(round(t2 - t, 2), 'Seconds to train SVC...')
        # Check the score of the SVC
        print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
        # Check the prediction time for a single sample
        t = time.time()

        image = mpimg.imread("../test_images/test1.jpg")
        draw_image = np.copy(image)

        # Uncomment the following line if you extracted training
        # data from .png images (scaled 0 to 1 by mpimg) and the
        # image you are searching is a .jpg (scaled 0 to 255)
        # image = image.astype(np.float32)/255

        windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
                               xy_window=(96, 96), xy_overlap=(0.5, 0.5))

        hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space,
                                     spatial_size=spatial_size, hist_bins=hist_bins,
                                     orient=orient, pix_per_cell=pix_per_cell,
                                     cell_per_block=cell_per_block,
                                     hog_channel=hog_channel, spatial_feat=spatial_feat,
                                     hist_feat=hist_feat, hog_feat=hog_feat)

        window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)

        plt.imshow(window_img)
        plt.savefig("../output_images/features_search/test1.png")

