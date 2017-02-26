import tensorflow as tf
tf.python.control_flow_ops = tf
import numpy as np
import cv2

from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Lambda
from keras.optimizers import Adam
from keras import backend as K
from scipy.ndimage.measurements import label


class VehicleDetector(object):
    def __init__(self, img_rows, img_cols, weights_file="model_segn_small_0p72.h5"):
        self.smooth = 1.0
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.model = self.get_small_unet(img_rows, img_cols)

        self.model.compile(optimizer=Adam(lr=1e-4),
                      loss=self.IOU_calc_loss, metrics=[self.IOU_calc])
        self.model.load_weights(weights_file)

        self.heatmap_prev = np.zeros((640, 960))
        self.heatmap_10 = [np.zeros((640, 960))] * 10

    def smooth_heatmap(self, heatmap):
        heatmap_10_1 = self.heatmap_10[1:]
        heatmap_10_1.append(heatmap)

        self.heatmap_10 = heatmap_10_1

        heatmap = np.mean(self.heatmap_10, axis=0)

        # heatmap = heatmap_prev*.2 + heatmap*.8
        # heatmap[heatmap>240] = 255
        # heatmap[heatmap<240] = 0

        return heatmap

    @staticmethod
    def get_small_unet(img_rows, img_cols):
        ## Redefining small U-net
        inputs = Input((img_rows, img_cols, 3))
        inputs_norm = Lambda(lambda x: x / 127.5 - 1.)
        conv1 = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(inputs)
        conv1 = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(pool1)
        conv2 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(pool2)
        conv3 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool3)
        conv4 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool4)
        conv5 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv5)

        up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=3)
        conv6 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up6)
        conv6 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv6)

        up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=3)
        conv7 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up7)
        conv7 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv7)

        up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=3)
        conv8 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(up8)
        conv8 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(conv8)

        up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=3)
        conv9 = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(up9)
        conv9 = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(conv9)

        conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

        model = Model(input=inputs, output=conv10)

        return model

    def IOU_calc(self, y_true, y_pred):
        # defining cost
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)

        return 2 * (intersection + self.smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + self.smooth)

    def IOU_calc_loss(self, y_true, y_pred):
        # defining cost
        return -self.IOU_calc(y_true, y_pred)

    @staticmethod
    def _draw_labeled_bboxes(img, labels):
        # Iterate through all detected cars
        for car_number in range(1, labels[1] + 1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            if ((np.max(nonzeroy) - np.min(nonzeroy) > 40) & (np.max(nonzerox) - np.min(nonzerox) > 40)):
                bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
                # Draw the box on the image
                print(bbox)
                cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
        # Return the image
        return img

    def _test_new_img(self, img):
        # Test Unet on new image
        img = cv2.resize(img, (self.img_cols, self.img_rows))
        img = np.reshape(img, (1, self.img_rows, self.img_cols, 3))
        pred = self.model.predict(img)
        return pred, img[0]

    def get_BB_new_img(self, img):
        # Get bounding boxes
        pred, img = self._test_new_img(img)
        img = np.array(img, dtype=np.uint8)
        img_pred = np.array(255 * pred[0], dtype=np.uint8)
        heatmap = img_pred[:, :, 0]
        heatmap = self.smooth_heatmap(heatmap)
        labels = label(heatmap)
        draw_img = self._draw_labeled_bboxes(np.copy(img), labels)
        return draw_img

    @staticmethod
    def get_labeled_bboxes(img, labels):
        # Get labeled boxex
        bbox_all = []
        for car_number in range(1, labels[1] + 1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])

            # Define a bounding box based on min/max x and y
            if ((np.max(nonzeroy) - np.min(nonzeroy) > 40) & (np.max(nonzerox) - np.min(nonzerox) > 40)):
                bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
                # Draw the box on the image
                # cv2.rectangle(img, bbox[0], bbox[1], (0,0,255),6)
                bbox_all.append(bbox)
        # Return the image
        return bbox_all

    def get_BB_new(self, img):
        # Take in RGB image
        pred, img = self._test_new_img(img)
        img = np.array(img, dtype=np.uint8)
        img_pred = np.array(255 * pred[0], dtype=np.uint8)
        heatmap = img_pred[:, :, 0]
        heatmap = self.smooth_heatmap(heatmap)
        # print(np.max(heatmap))
        heatmap[heatmap > 240] = 255
        heatmap[heatmap <= 240] = 0
        labels = label(heatmap)

        bbox_all = self.get_labeled_bboxes(np.copy(img), labels)
        return bbox_all

    def get_Unet_mask(self, img):
        # Take in RGB image
        pred, img = self._test_new_img(img)
        img = np.array(img, dtype=np.uint8)
        img_pred = np.array(255 * pred[0], dtype=np.uint8)
        heatmap = img_pred[:, :, 0]
        heatmap = self.smooth_heatmap(heatmap)
        # labels = label(heatmap)
        return self.stack_arr(heatmap)

    @staticmethod
    def stack_arr(arr):
        return np.stack((arr, arr, arr), axis=2)

