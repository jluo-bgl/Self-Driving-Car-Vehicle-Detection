import numpy as np
import cv2
import scipy.ndimage
from data_load import FeedingData


def image_itself(feeding_data):
    return feeding_data.image(), feeding_data.steering_angle


def shift_image_generator(angle_offset_pre_pixel=0.003):
    def _generator(feeding_data):
        image, angle, _ = _shift_image(
            feeding_data.image(), feeding_data.steering_angle, 100, 20, angle_offset_pre_pixel=angle_offset_pre_pixel)
        return image, angle

    return _generator


def brightness_image_generator(brightness_range=0.25):
    def _generator(feeding_data):
        img = feeding_data.image()
        # Convert the image to HSV
        temp = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        # Compute a random brightness value and apply to the image
        brightness = brightness_range + np.random.uniform()
        temp[:, :, 2] = temp[:, :, 2] * brightness

        # Convert back to RGB and return
        return cv2.cvtColor(temp, cv2.COLOR_HSV2RGB), feeding_data.steering_angle

    return _generator


def shadow_generator(feeding_data):
    image = feeding_data.image()
    top_y = image.shape[1] * np.random.uniform()
    top_x = 0
    bot_x = image.shape[0]
    bot_y = image.shape[1] * np.random.uniform()
    image_hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    shadow_mask = 0 * image_hls[:, :, 1]
    X_m = np.mgrid[0:image.shape[0], 0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0], 0:image.shape[1]][1]

    shadow_mask[((X_m - top_x) * (bot_y - top_y) - (bot_x - top_x) * (Y_m - top_y) >= 0)] = 1
    if np.random.randint(2) == 1:
        random_bright = .5
        cond1 = shadow_mask == 1
        cond0 = shadow_mask == 0
        if np.random.randint(2) == 1:
            image_hls[:, :, 1][cond1] = image_hls[:, :, 1][cond1] * random_bright
        else:
            image_hls[:, :, 1][cond0] = image_hls[:, :, 1][cond0] * random_bright
    image = cv2.cvtColor(image_hls, cv2.COLOR_HLS2RGB)
    return image, feeding_data.steering_angle


def random_generators(*generators):
    def _generator(feeding_data):
        index = np.random.randint(0, len(generators))
        return generators[index](feeding_data)

    return _generator


def flip_generator(feeding_data):
    image, angle = feeding_data.image(), feeding_data.steering_angle
    return cv2.flip(image, 1), -angle


def pipe_line_generators(*generators):
    """
    pipe line of generators, generator will run one by one
    :param generators:
    :return:
    """
    def _generator(feeding_data):
        intermediary_feeding_data = feeding_data
        for generator in generators:
            image, angle = generator(intermediary_feeding_data)
            intermediary_feeding_data = FeedingData(image, angle)
        return intermediary_feeding_data.image(), intermediary_feeding_data.steering_angle

    return _generator


def pipe_line_random_generators(*generators):
    def _generator(feeding_data):
        count = np.random.randint(0, len(generators)+1)
        intermediary_feeding_data = feeding_data
        for index in range(count):
            generator = generators[index]
            image, angle = generator(intermediary_feeding_data)
            intermediary_feeding_data = FeedingData(image, angle)
        return intermediary_feeding_data.image(), intermediary_feeding_data.steering_angle

    return _generator


def filter_generator(generator, angle_threshold=0.1):
    def _generator(feeding_data):
        image, angle = None, None
        for index in range(20):
            if angle is None or angle <= angle_threshold:
                image, angle = generator(feeding_data)
            else:
                break

        return image, angle

    return _generator


def _shift_image(image, steer, left_right_shift_range, top_bottom_shift_range, angle_offset_pre_pixel=0.003):
    shift_size = round(left_right_shift_range * np.random.uniform(-0.5, 0.5))
    steer_ang = steer + shift_size * angle_offset_pre_pixel
    top_bottom_shift_size = round(top_bottom_shift_range * np.random.uniform(-0.5, 0.5))
    if shift_size >= image.shape[1]:
        image_tr = image
        # print("WARNING Image is smaller then shift size, original image returned")
    else:
        image_tr = scipy.ndimage.interpolation.shift(image, (top_bottom_shift_size, shift_size, 0))
    return image_tr, steer_ang, shift_size


