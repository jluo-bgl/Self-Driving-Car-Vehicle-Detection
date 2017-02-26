import argparse
import colorsys
import imghdr
import os
import random

import numpy as np
from keras import backend as K
from keras.models import load_model
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import VideoFileClip

from yolo.yad2k.models.keras_yolo import yolo_eval, yolo_head


class YoloDetector(object):
    def __init__(self,
                 model_path="yolo/model_data/yolo.h5",
                 anchors_path="yolo/model_data/yolo_anchors.txt",
                 classes_path="yolo/model_data/coco_classes.txt",
                 font_file_name="yolo/font/FiraMono-Medium.otf"):
        assert model_path.endswith('.h5'), 'Keras model must be a .h5 file.'

        self.model_path = model_path
        self.class_names = self._read_class_names(classes_path)
        self.colors = self._generator_colors(self.class_names)
        self.font_file_name = font_file_name

        self.sess = K.get_session()
        self.yolo_model = load_model(model_path)

        anchors = self._read_anchors(anchors_path)
        self._validate_model_and_data(self.class_names, anchors, self.yolo_model)

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Check if model is fully convolutional, assuming channel last order.
        self.model_image_size = self.yolo_model.layers[0].input_shape[1:3]
        self.is_fixed_size = self.model_image_size != (None, None)

        # Generate output tensor targets for filtered bounding boxes.
        yolo_outputs = yolo_head(self.yolo_model.output, anchors, len(self.class_names))
        self.input_image_shape = K.placeholder(shape=(2,))
        boxes, scores, classes = yolo_eval(
            yolo_outputs,
            self.input_image_shape,
            score_threshold=0.3,
            iou_threshold=0.3)
        self.boxes = boxes
        self.scores = scores
        self.classes = classes

    @staticmethod
    def _validate_model_and_data(class_names, anchors, yolo_model):
        num_classes = len(class_names)
        num_anchors = len(anchors)
        # TODO: Assumes dim ordering is channel last
        model_output_channels = yolo_model.layers[-1].output_shape[-1]
        assert model_output_channels == num_anchors * (num_classes + 5), \
            'Mismatch between model and given anchor and class sizes. ' \
            'Specify matching anchors and classes with --anchors_path and ' \
            '--classes_path flags.'

    @staticmethod
    def _generator_colors(class_names):
        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(class_names), 1., 1.)
                      for x in range(len(class_names))]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                colors))
        random.seed(10101)  # Fixed seed for consistent colors across runs.
        random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
        random.seed(None)  # Reset seed to default.
        return colors

    @staticmethod
    def _read_class_names(classes_path):
        with open(classes_path) as f:
            class_names = f.readlines()

        class_names = [c.strip() for c in class_names]
        return class_names

    @staticmethod
    def _read_anchors(anchors_path):
        with open(anchors_path) as f:
            anchors = f.readline()
            anchors = [float(x) for x in anchors.split(',')]
            anchors = np.array(anchors).reshape(-1, 2)
            return anchors

    def predict(self, image_pil):
        if self.is_fixed_size:
            resized_image = image_pil.resize(
                tuple(reversed(self.model_image_size)), Image.BICUBIC)
            image_data = np.array(resized_image, dtype='float32')
        else:
            image_data = np.array(image_pil, dtype='float32')

        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image_pil.size[1], image_pil.size[0]],
                K.learning_phase(): 0
            })

        return out_boxes, out_scores, out_classes

    def draw_border_boxes(self, image, out_boxes, out_scores, out_classes):
        font = ImageFont.truetype(
            font=self.font_file_name,
            size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)

            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

    def process_image_array(self, image_array):
        image = Image.fromarray(image_array)
        image = self.process_image_pil(image)
        return np.array(image)

    def process_image_pil(self, image_pil):
        out_boxes, out_scores, out_classes = self.predict(image_pil)
        self.draw_border_boxes(image_pil, out_boxes, out_scores, out_classes)
        return image_pil

    def process_image_file(self, input_image_file_name, output_image_filename):
        image = Image.open(input_image_file_name)
        image = self.process_image_pil(image)
        image.save(output_image_filename, quality=90)

    def process_folder(self, input_folder, output_folder):
        for image_file in os.listdir(input_folder):
            try:
                image_type = imghdr.what(os.path.join(input_folder, image_file))
                if not image_type:
                    continue
            except IsADirectoryError:
                continue

            self.process_image_file(os.path.join(input_folder, image_file), os.path.join(output_folder, image_file))

    def shutdown(self):
        self.sess.close()


if __name__ == "__main__":

    def remove_mp4_extension(file_name):
        return file_name.replace(".mp4", "")

    def process_folder(yolo):
        yolo.process_folder(input_folder="test_images", output_folder="output_images/object-detect")

    def process_video(yolo):
        video_file = 'back_home.mov'
        # video_file = 'project_video.mp4'
        # video_file = 'challenge_video.mp4'
        clip = VideoFileClip(video_file, audio=False)
        t_start = 127
        t_end = 147
        if t_end > 0.0:
            clip = clip.subclip(t_start=t_start, t_end=t_end)
        else:
            clip = clip.subclip(t_start=t_start)

        clip = clip.fl_image(yolo.process_image_array)
        clip.write_videofile("{}_output_detect.mp4".format(remove_mp4_extension(video_file)), audio=False)

    yolo = YoloDetector()
    # process_folder(yolo)
    process_video(yolo)
    yolo.shutdown()

