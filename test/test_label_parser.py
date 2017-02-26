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


import pandas as pd

dir_label = ['../data/object-dataset',
             'object-detection-crowdai']

class LabelParseTest(unittest.TestCase):

    def test_parse_label_csv_file(self):
        df_files2 = pd.read_csv(dir_label[0] + '/labels.csv', header=None, sep=' ')

        df_files2.columns = ['Frame', 'xmin', 'xmax', 'ymin', 'ymax', 'ind', 'Label', 'RM']
        df_vehicles2 = df_files2[(df_files2['Label'] == 'car') | (df_files2['Label'] == 'truck')].reset_index()
        df_vehicles2 = df_vehicles2.drop('index', 1)
        df_vehicles2 = df_vehicles2.drop('RM', 1)
        df_vehicles2 = df_vehicles2.drop('ind', 1)

        df_vehicles2['File_Path'] = dir_label[0] + '/' + df_vehicles2['Frame']

        print(df_vehicles2.head())

