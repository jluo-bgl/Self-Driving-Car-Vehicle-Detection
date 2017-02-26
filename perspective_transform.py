import cv2
import numpy as np


def calculate_transform_matrices(image_width, image_height):
    bottomW = image_width
    topW = 249
    bottomH = image_height - 20
    topH = bottomH - 228
    region_vertices = np.array([[((image_width - bottomW) // 2, bottomH),
                                 ((image_width - topW) // 2, topH),
                                 ((image_width + topW) // 2, topH),
                                 ((image_width + bottomW) // 2, bottomH)]])
    offsetH = 10
    offsetW = 100
    dest_vertices = np.array([[(offsetW, image_height - offsetH),
                               (offsetW, offsetH),
                               (image_width - offsetW, offsetH),
                               (image_width - offsetW, image_height - offsetH)]])

    perspective_transform_matrix = cv2.getPerspectiveTransform(
        np.float32(region_vertices), np.float32(dest_vertices))
    inversion_perspective_transform_matrix = cv2.getPerspectiveTransform(
        np.float32(dest_vertices), np.float32(region_vertices))

    return perspective_transform_matrix, inversion_perspective_transform_matrix


def perspective_transform(img, perspective_transform_matrix):
    return cv2.warpPerspective(img, perspective_transform_matrix, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)


def inversion_perspective_transform(img, inversion_perspective_transform_matrix):
    return cv2.warpPerspective(img, inversion_perspective_transform_matrix, (img.shape[1], img.shape[0]),
                               flags=cv2.INTER_LINEAR)
