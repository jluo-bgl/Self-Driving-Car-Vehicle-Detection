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

