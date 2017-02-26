### Import libraries

import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import glob

from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Lambda
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from scipy.ndimage.measurements import label
import time

dir_label = ['object-dataset',
             'object-detection-crowdai']

import pandas as pd

df_files1 = pd.read_csv(dir_label[1] + '/labels.csv', header=0)
df_vehicles1 = df_files1[(df_files1['Label'] == 'Car') | (df_files1['Label'] == 'Truck')].reset_index()
df_vehicles1 = df_vehicles1.drop('index', 1)
df_vehicles1['File_Path'] = dir_label[1] + '/' + df_vehicles1['Frame']
df_vehicles1 = df_vehicles1.drop('Preview URL', 1)
print(dir_label[1])
df_vehicles1.head()

df_files2 = pd.read_csv('object-dataset/labels.csv', header=None)
df_files2.columns = ['Frame', 'xmin', 'xmax', 'ymin', 'ymax', 'ind', 'Label', 'RM']
df_vehicles2 = df_files2[(df_files2['Label'] == 'car') | (df_files2['Label'] == 'truck')].reset_index()
df_vehicles2 = df_vehicles2.drop('index', 1)
df_vehicles2 = df_vehicles2.drop('RM', 1)
df_vehicles2 = df_vehicles2.drop('ind', 1)

df_vehicles2['File_Path'] = dir_label[0] + '/' + df_vehicles2['Frame']

df_vehicles2.head()

df_vehicles = pd.concat([df_vehicles1, df_vehicles2]).reset_index()
df_vehicles = df_vehicles.drop('index', 1)
df_vehicles.columns = ['File_Path', 'Frame', 'Label', 'ymin', 'xmin', 'ymax', 'xmax']
df_vehicles.head()

trans_range = 0


### Augmentation functions

def augment_brightness_camera_images(image):
    ### Augment brightness
    image1 = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    random_bright = .25 + np.random.uniform()
    # print(random_bright)
    image1[:, :, 2] = image1[:, :, 2] * random_bright
    image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)
    return image1


def trans_image(image, bb_boxes_f, trans_range):
    # Translation augmentation
    bb_boxes_f = bb_boxes_f.copy(deep=True)

    tr_x = trans_range * np.random.uniform() - trans_range / 2
    tr_y = trans_range * np.random.uniform() - trans_range / 2

    Trans_M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])
    rows, cols, channels = image.shape
    bb_boxes_f['xmin'] = bb_boxes_f['xmin'] + tr_x
    bb_boxes_f['xmax'] = bb_boxes_f['xmax'] + tr_x
    bb_boxes_f['ymin'] = bb_boxes_f['ymin'] + tr_y
    bb_boxes_f['ymax'] = bb_boxes_f['ymax'] + tr_y

    image_tr = cv2.warpAffine(image, Trans_M, (cols, rows))

    return image_tr, bb_boxes_f


def stretch_image(img, bb_boxes_f, scale_range):
    # Stretching augmentation

    bb_boxes_f = bb_boxes_f.copy(deep=True)

    tr_x1 = scale_range * np.random.uniform()
    tr_y1 = scale_range * np.random.uniform()
    p1 = (tr_x1, tr_y1)
    tr_x2 = scale_range * np.random.uniform()
    tr_y2 = scale_range * np.random.uniform()
    p2 = (img.shape[1] - tr_x2, tr_y1)

    p3 = (img.shape[1] - tr_x2, img.shape[0] - tr_y2)
    p4 = (tr_x1, img.shape[0] - tr_y2)

    pts1 = np.float32([[p1[0], p1[1]],
                       [p2[0], p2[1]],
                       [p3[0], p3[1]],
                       [p4[0], p4[1]]])
    pts2 = np.float32([[0, 0],
                       [img.shape[1], 0],
                       [img.shape[1], img.shape[0]],
                       [0, img.shape[0]]]
                      )

    M = cv2.getPerspectiveTransform(pts1, pts2)
    img = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))
    img = np.array(img, dtype=np.uint8)

    bb_boxes_f['xmin'] = (bb_boxes_f['xmin'] - p1[0]) / (p2[0] - p1[0]) * img.shape[1]
    bb_boxes_f['xmax'] = (bb_boxes_f['xmax'] - p1[0]) / (p2[0] - p1[0]) * img.shape[1]
    bb_boxes_f['ymin'] = (bb_boxes_f['ymin'] - p1[1]) / (p3[1] - p1[1]) * img.shape[0]
    bb_boxes_f['ymax'] = (bb_boxes_f['ymax'] - p1[1]) / (p3[1] - p1[1]) * img.shape[0]

    return img, bb_boxes_f


def get_image_name(df, ind, size=(640, 300), augmentation=False, trans_range=20, scale_range=20):
    ### Get image by name

    file_name = df['File_Path'][ind]
    img = cv2.imread(file_name)
    img_size = np.shape(img)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    name_str = file_name.split('/')
    name_str = name_str[-1]
    # print(name_str)
    # print(file_name)
    bb_boxes = df[df['Frame'] == name_str].reset_index()
    img_size_post = np.shape(img)

    if augmentation == True:
        img, bb_boxes = trans_image(img, bb_boxes, trans_range)
        img, bb_boxes = stretch_image(img, bb_boxes, scale_range)
        img = augment_brightness_camera_images(img)

    bb_boxes['xmin'] = np.round(bb_boxes['xmin'] / img_size[1] * img_size_post[1])
    bb_boxes['xmax'] = np.round(bb_boxes['xmax'] / img_size[1] * img_size_post[1])
    bb_boxes['ymin'] = np.round(bb_boxes['ymin'] / img_size[0] * img_size_post[0])
    bb_boxes['ymax'] = np.round(bb_boxes['ymax'] / img_size[0] * img_size_post[0])
    bb_boxes['Area'] = (bb_boxes['xmax'] - bb_boxes['xmin']) * (bb_boxes['ymax'] - bb_boxes['ymin'])
    # bb_boxes = bb_boxes[bb_boxes['Area']>400]


    return name_str, img, bb_boxes


def get_mask_seg(img, bb_boxes_f):
    #### Get mask

    img_mask = np.zeros_like(img[:, :, 0])
    for i in range(len(bb_boxes_f)):
        # plot_bbox(bb_boxes,i,'g')
        bb_box_i = [bb_boxes_f.iloc[i]['xmin'], bb_boxes_f.iloc[i]['ymin'],
                    bb_boxes_f.iloc[i]['xmax'], bb_boxes_f.iloc[i]['ymax']]
        img_mask[bb_box_i[1]:bb_box_i[3], bb_box_i[0]:bb_box_i[2]] = 1.
        img_mask = np.reshape(img_mask, (np.shape(img_mask)[0], np.shape(img_mask)[1], 1))
    return img_mask


def plot_im_mask(im, im_mask):
    ### Function to plot image mask

    im = np.array(im, dtype=np.uint8)
    im_mask = np.array(im_mask, dtype=np.uint8)
    plt.subplot(1, 3, 1)
    plt.imshow(im)
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(im_mask[:, :, 0])
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.bitwise_and(im, im, mask=im_mask));
    plt.axis('off')
    plt.show();


def plot_bbox(bb_boxes, ind_bb, color='r', linewidth=2):
    ### Plot bounding box

    bb_box_i = [bb_boxes.iloc[ind_bb]['xmin'],
                bb_boxes.iloc[ind_bb]['ymin'],
                bb_boxes.iloc[ind_bb]['xmax'],
                bb_boxes.iloc[ind_bb]['ymax']]
    plt.plot([bb_box_i[0], bb_box_i[2], bb_box_i[2],
              bb_box_i[0], bb_box_i[0]],
             [bb_box_i[1], bb_box_i[1], bb_box_i[3],
              bb_box_i[3], bb_box_i[1]],
             color, linewidth=linewidth)


def plot_im_bbox(im, bb_boxes):
    ### Plot image and bounding box
    plt.imshow(im)
    for i in range(len(bb_boxes)):
        plot_bbox(bb_boxes, i, 'g')

        bb_box_i = [bb_boxes.iloc[i]['xmin'], bb_boxes.iloc[i]['ymin'],
                    bb_boxes.iloc[i]['xmax'], bb_boxes.iloc[i]['ymax']]
        plt.plot(bb_box_i[0], bb_box_i[1], 'rs')
        plt.plot(bb_box_i[2], bb_box_i[3], 'bs')
    plt.axis('off');


#### Test translation and stretching augmentations

name_str, img, bb_boxes = get_image_name(df_vehicles, 1, augmentation=False, trans_range=0, scale_range=0)
img_mask = get_mask_seg(img, bb_boxes)

tr_x1 = 80
tr_y1 = 30
tr_x2 = 40
tr_y2 = 20

p1 = (tr_x1, tr_y1)
p2 = (img.shape[1] - tr_x2, tr_y1)

p3 = (img.shape[1] - tr_x2, img.shape[0] - tr_y2)
p4 = (tr_x1, img.shape[0] - tr_y2)

pts1 = np.float32([[p1[0], p1[1]],
                   [p2[0], p2[1]],
                   [p3[0], p3[1]],
                   [p4[0], p4[1]]])
pts2 = np.float32([[0, 0],
                   [img.shape[1], 0],
                   [img.shape[1], img.shape[0]], [0, img.shape[0]]]
                  )

M = cv2.getPerspectiveTransform(pts1, pts2)
dst = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))
dst = np.array(dst, dtype=np.uint8)

plt.figure(figsize=(12, 8))
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.plot(p1[0], p1[1], 'mo')
plt.plot(p2[0], p2[1], 'mo')
plt.plot(p3[0], p3[1], 'mo')
plt.plot(p4[0], p4[1], 'mo')
for i in range(len(bb_boxes)):
    plot_bbox(bb_boxes, i, 'g')

    bb_box_i = [bb_boxes.iloc[i]['xmin'], bb_boxes.iloc[i]['ymin'],
                bb_boxes.iloc[i]['xmax'], bb_boxes.iloc[i]['ymax']]
    plt.plot(bb_box_i[0], bb_box_i[1], 'rs')
    plt.plot(bb_box_i[2], bb_box_i[3], 'bs')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(dst)
bb_boxes1 = bb_boxes.copy(deep=True)
bb_boxes1['xmin'] = (bb_boxes['xmin'] - p1[0]) / (p2[0] - p1[0]) * img.shape[1]
bb_boxes1['xmax'] = (bb_boxes['xmax'] - p1[0]) / (p2[0] - p1[0]) * img.shape[1]
bb_boxes1['ymin'] = (bb_boxes['ymin'] - p1[1]) / (p3[1] - p1[1]) * img.shape[0]
bb_boxes1['ymax'] = (bb_boxes['ymax'] - p1[1]) / (p3[1] - p1[1]) * img.shape[0]
plt.plot(0, 0, 'mo')
plt.plot(img.shape[1], 0, 'mo')
plt.plot(img.shape[1], img.shape[0], 'mo')
plt.plot(0, img.shape[0], 'mo')
plot_im_bbox(dst, bb_boxes1)

plt.axis('off');

#### Test translation and stretching augmentations

name_str, img, bb_boxes = get_image_name(df_vehicles, 1, augmentation=False)
img_mask = get_mask_seg(img, bb_boxes)

plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plot_im_bbox(img, bb_boxes)

plt.subplot(2, 2, 2)
plt.imshow(img_mask[:, :, 0])
plt.axis('off')

plt.subplot(2, 2, 3)
# bb_boxes1 = bb_boxes.copy()
dst, bb_boxes1 = stretch_image(img, bb_boxes, 100)

plt.imshow(dst)

plot_im_bbox(dst, bb_boxes1)

plt.subplot(2, 2, 4)
img_mask2 = get_mask_seg(dst, bb_boxes1)
plt.imshow(img_mask2[:, :, 0])
plt.axis('off');

#### Test translation and stretching augmentations

name_str, img, bb_boxes = get_image_name(df_vehicles, 200, augmentation=False)
img_mask = get_mask_seg(img, bb_boxes)

plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plot_im_bbox(img, bb_boxes)

plt.subplot(2, 2, 2)
plt.imshow(img_mask[:, :, 0])
plt.axis('off')

plt.subplot(2, 2, 3)
# bb_boxes1 = bb_boxes.copy()
img_trans, bb_boxes1 = trans_image(img, bb_boxes, 100)

plt.imshow(img_trans)

plot_im_bbox(img_trans, bb_boxes1)
img_mask2 = get_mask_seg(img_trans, bb_boxes1)

plt.subplot(2, 2, 4)
plt.imshow(img_mask2[:, :, 0])
plt.axis('off');

plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plot_im_bbox(img, bb_boxes)
plt.subplot(2, 2, 2)
# bb_boxes1 = bb_boxes.copy()
img_trans, bb_boxes1 = trans_image(img, bb_boxes, 50)
plt.imshow(img_trans)
plot_im_bbox(img_trans, bb_boxes1)

#### Put all the augmentations in 1 function with a flag for augmentation


name_str, img, bb_boxes = get_image_name(df_vehicles, 1, augmentation=False)
img_mask = get_mask_seg(img, bb_boxes)

plt.figure(figsize=(6, 4))
plt.imshow(img)
plot_im_bbox(img, bb_boxes)
plt.show()

plot_im_mask(img, img_mask)


#### Training generator, generates augmented images
def generate_train_batch(data, batch_size=32):
    batch_images = np.zeros((batch_size, img_rows, img_cols, 3))
    batch_masks = np.zeros((batch_size, img_rows, img_cols, 1))
    while 1:
        for i_batch in range(batch_size):
            i_line = np.random.randint(len(data) - 2000)
            name_str, img, bb_boxes = get_image_name(df_vehicles, i_line,
                                                     size=(img_cols, img_rows),
                                                     augmentation=True,
                                                     trans_range=50,
                                                     scale_range=50
                                                     )
            img_mask = get_mask_seg(img, bb_boxes)
            batch_images[i_batch] = img
            batch_masks[i_batch] = img_mask
        yield batch_images, batch_masks


#### Testing generator, generates augmented images
def generate_test_batch(data, batch_size=32):
    batch_images = np.zeros((batch_size, img_rows, img_cols, 3))
    batch_masks = np.zeros((batch_size, img_rows, img_cols, 1))
    while 1:
        for i_batch in range(batch_size):
            i_line = np.random.randint(2000)
            i_line = i_line + len(data) - 2000
            name_str, img, bb_boxes = get_image_name(df_vehicles, i_line,
                                                     size=(img_cols, img_rows),
                                                     augmentation=False,
                                                     trans_range=0,
                                                     scale_range=0
                                                     )
            img_mask = get_mask_seg(img, bb_boxes)
            batch_images[i_batch] = img
            batch_masks[i_batch] = img_mask
        yield batch_images, batch_masks


##### Image size,
img_rows = 640
img_cols = 960

##### Testing the generator


training_gen = generate_train_batch(df_vehicles, 10)

batch_img, batch_mask = next(training_gen)

### Plotting generator output
for i in range(10):
    im = np.array(batch_img[i], dtype=np.uint8)
    im_mask = np.array(batch_mask[i], dtype=np.uint8)
    plt.subplot(1, 3, 1)
    plt.imshow(im)
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(im_mask[:, :, 0])
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.bitwise_and(im, im, mask=im_mask));
    plt.axis('off')
    plt.show();


### IOU or dice coeff calculation

def IOU_calc(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)

    return 2 * (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def IOU_calc_loss(y_true, y_pred):
    return -IOU_calc(y_true, y_pred)


### Defining a small Unet
### Smaller Unet defined so it fits in memory

def get_small_unet():
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


### Generator

training_gen = generate_train_batch(df_vehicles, 1)
smooth = 1.
model = get_small_unet()
model.compile(optimizer=Adam(lr=1e-4),
              loss=IOU_calc_loss, metrics=[IOU_calc])

### Using previously trained data. Set load_pretrained = False, increase epochs and train for full training.
load_pretrained = True
if load_pretrained == True:
    model.load_weights("model_segn_small_0p72.h5")

history = model.fit_generator(training_gen,
                              samples_per_epoch=1000,
                              nb_epoch=1)

model.save('model_detect_SmallUnet.h5')

### Save weights
model.save_weights("model_segn_small_udacity_0p71.h5", overwrite=True)

### Testing generator

testing_gen = generate_test_batch(df_vehicles, 20)

import time

start = time.time()

pred_all = model.predict(batch_img)
end = time.time()
end - start

### Test on last frames of data

batch_img, batch_mask = next(testing_gen)
pred_all = model.predict(batch_img)
np.shape(pred_all)

for i in range(20):
    im = np.array(batch_img[i], dtype=np.uint8)
    im_mask = np.array(255 * batch_mask[i], dtype=np.uint8)
    im_pred = np.array(255 * pred_all[i], dtype=np.uint8)

    rgb_mask_pred = cv2.cvtColor(im_pred, cv2.COLOR_GRAY2RGB)
    rgb_mask_pred[:, :, 1:3] = 0 * rgb_mask_pred[:, :, 1:2]
    rgb_mask_true = cv2.cvtColor(im_mask, cv2.COLOR_GRAY2RGB)
    rgb_mask_true[:, :, 0] = 0 * rgb_mask_true[:, :, 0]
    rgb_mask_true[:, :, 2] = 0 * rgb_mask_true[:, :, 2]

    img_pred = cv2.addWeighted(rgb_mask_pred, 0.5, im, 0.5, 0)
    img_true = cv2.addWeighted(rgb_mask_true, 0.5, im, 0.5, 0)

    plt.figure(figsize=(8, 3))
    plt.subplot(1, 3, 1)
    plt.imshow(im)
    plt.title('Original image')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(img_pred)
    plt.title('Predicted segmentation mask')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(img_true)
    plt.title('Ground truth BB')
    plt.axis('off')
    plt.show()


