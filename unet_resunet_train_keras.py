from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import time
import sys
from itertools import product
import cv2
import os
import matplotlib.pyplot as plt
from libtiff import TIFF
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D #to add convolution layers
from tensorflow.keras.layers import MaxPooling2D # to add pooling layers
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import concatenate
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator




start = time.process_time()
from tensorflow.keras import backend as K
def iou(y_true, y_pred, smooth = 100):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    union = K.sum(y_true,-1) + K.sum(y_pred,-1) - intersection
    #sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    iou_acc = (intersection + smooth) / (union + smooth)
    return iou_acc


# In[5]:

def jaccard_coef(y_true, y_pred):
    # __author__ = Vladimir Iglovikov
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)

def jaccard_coef_int(y_true, y_pred):
    # __author__ = Vladimir Iglovikov
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))

    intersection = K.sum(y_true * y_pred_pos, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.mean(jac)

# In[6]:

smooth = 1.

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

# In[6]:

def weighted_binary_crossentropy(y_true, y_pred):
    class_loglosses = K.mean(K.binary_crossentropy(y_true, y_pred), axis=[0, 1, 2])
    return K.sum(class_loglosses * K.constant(class_weights))

# In[7]:
# To read the images in numerical order
import re
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

# In[]:

def mean_normalize_dummy(image):
  std_img = np.std(image)
  mean_img = np.mean(image)
  norm_img = image
  return std_img, mean_img, norm_img

def mean_normalize_org(image):
  std_img = np.std(image)
  mean_img = np.mean(image)
  norm_img = (image - mean_img)/std_img
  return std_img, mean_img, norm_img

def mean_normalize(image):
  std_lst = []
  mean_lst = []
  for idx in range(image.shape[2]):
    std_chan = np.std(image[:,:,idx])
    mean_chan = np.mean(image[:,:,idx])
    std_lst.append(std_chan)
    mean_lst.append(mean_chan)

  std_img = np.array([std_lst])
  mean_img = np.array([mean_lst])
  norm_img = (image - mean_img)/std_img
  return std_img, mean_img, norm_img

def reverse_mean_normalize(norm, std_img, mean_img):
  image = (norm * std_img) + mean_img
  return image

def reverse_mean_normalize_dummy(norm, std_img, mean_img):
  return norm

IMAGE_SIZE = 256

imgname_list = os.listdir('/home/rcgs/ML_38/AerialImageDataset/train/images/')
imgname_list = sorted(imgname_list)

mask_list = os.listdir('/home/rcgs/ML_38/AerialImageDataset/train/gt/')
mask_list = sorted(mask_list)

Num_Epochs = 1
tile_sizex = IMAGE_SIZE
tile_sizey = IMAGE_SIZE

save_logs = True
show_graphs = True
save_tiles = False
view_tiles = False
augment_images = False
use_callbacks = False
#load_model_weights = True # first step config
load_model_weights = False # second step config
lr_find = False
use_clr = False

#IMG_FILL_VALUE = 255
#init_learning_rate = 1e-2 # trial with SGD
init_learning_rate = 1e-3 # first step learning rate
#init_learning_rate = 1e-5 # second step learning rate
min_feature_cnt = 1
train_drop_rate = 0.2
train_batch_size = 16
start_epoch = 0
save_weights_periodicity = 10
CLR_MIN_LR = 1e-5
CLR_MAX_LR = 1e-2
CLR_STEP_SIZE = 8
CLR_METHOD = "triangular2" #"triangular"/"triangular2"/"exp_range"
lr_schedule = None #None/"step"/"linear"/"poly"

print(len(imgname_list))
print(len(mask_list))

i=1

imageseg = dict()

img_folder_path = '/home/rcgs/ML_38/AerialImageDataset/train/images/'
for img_name in imgname_list:
  print(img_name)

  img_name = img_folder_path+img_name
  tifinimg = TIFF.open(img_name)
  inimg = tifinimg.read_image()
  TIFF.close(tifinimg)

  img_dtype = type(inimg[0,0,0])

  print(inimg.shape)

  (H,W,Ch) = inimg.shape

  (H_delta,W_delta) = 0,0

  if (H % tile_sizey) != 0:
    H_delta = tile_sizey - (H % tile_sizey)
    #print(H_delta)
  if (W % tile_sizex) != 0:
    W_delta = tile_sizex - (W % tile_sizex)
    #print(W_delta)

  #print(img_dtype)

  IMG_FILL_VALUE = np.max(inimg)

  top, bottom, left, right = 0, H_delta, 0, W_delta

  inimg_new =  np.ones((H+H_delta, W+W_delta,Ch), dtype=img_dtype) * IMG_FILL_VALUE
  inimg_new[:H,:W,:] = inimg

  instd_img, inmean_img, inimg_new = mean_normalize(inimg_new)

  offsets = product(range(0, W, tile_sizex), range(0, H, tile_sizey))

  cnt = 1
  view_tiles_img = view_tiles

  for row_off,col_off in offsets:
    print("Img:{}. (Row:Col) = ({}:{})".format(cnt,row_off,col_off))

    col_start, col_end, row_start, row_end = col_off, col_off+tile_sizey-1, row_off, row_off+tile_sizex-1

    imgtile = inimg_new[col_start:col_end+1,row_start:row_end+1,:]
    imageseg['{}-{}-{}'.format(i, row_off,col_off)] = imgtile

    cnt = cnt + 1

  i=i+1

  if i>10:
    break
    #print(imgtile.shape)

len(imageseg)

i=1

imageseg2 = dict()
trainx_list = []
trainy_list = []

mask_folder_path = '/home/rcgs/ML_38/AerialImageDataset/train/gt/'
for mask_name in mask_list:
  print(mask_name)

  img_name = mask_folder_path+mask_name
  tifinimg = TIFF.open(img_name)
  inmask = tifinimg.read_image()
  TIFF.close(tifinimg)

  mask_dtype = type(inmask[0,0])

  print(mask_dtype.shape)

  (H2,W2) = inmask.shape

  (H_delta2,W_delta2) = 0,0

  if (H2 % tile_sizey) != 0:
    H_delta2 = tile_sizey - (H2 % tile_sizey)
  if (W2 % tile_sizex) != 0:
    W_delta2 = tile_sizex - (W2 % tile_sizex)

  top2, bottom2, left2, right2 = 0, H_delta2, 0, W_delta2
  inmask_new =  cv2.copyMakeBorder(inmask, top2, bottom2, left2, right2, cv2.BORDER_CONSTANT, value=0)
    #print(W_delta)
  mask_minval = np.min(inmask_new)
  mask_maxval = np.max(inmask_new)
  inmask_new = inmask_new / (mask_maxval - mask_minval)

  offsets2 = product(range(0, W2, tile_sizex), range(0, H2, tile_sizey))

  cnt2 = 1
  #print(img_dtype)
  for row_off2,col_off2 in offsets2:
    print("Mask:{}. (Row:Col) = ({}:{})".format(cnt2,row_off2,col_off2))
    col_start2, col_end2, row_start2, row_end2 = col_off2, col_off2+tile_sizey-1, row_off2, row_off2+tile_sizex-1
    imgtile2 = inmask_new[col_start2:col_end2+1,row_start2:row_end2+1]

    imgtile2_org = imgtile2.copy()

    imgtile2 = np.expand_dims(imgtile2, axis=2)
     # shape (1, x_pixels, y_pixels, n_bands)
    imageseg2['{}-{}-{}'.format(i, row_off2,col_off2)] = imgtile2

    if (cv2.countNonZero(imageseg2['{}-{}-{}'.format(i, row_off2,col_off2)])) > min_feature_cnt:

      trainx_list.append(imageseg['{}-{}-{}'.format(i, row_off2,col_off2)])

      trainy_list.append(imgtile2)

  cnt2 = cnt2 + 1

  i=i+1

  if i>10:
    break
    #print(imgtile.shape)



trainx = np.asarray(trainx_list)
trainy = np.asarray(trainy_list)
print(trainx.shape)
print(trainy.shape)


min_samples = min(trainx.shape[0],trainy.shape[0])

print(min_samples)

x_train, x_test, y_train, y_test = train_test_split(trainx[:min_samples], trainy[:min_samples], test_size=0.20, random_state=4)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
print(x_train.shape[1:])

inputs = Input(trainx.shape[1:])
n_classes=1
im_sz=Image_size
n_channels=Ch
n_filters_start=32
growth_factor=2
upconv=True
class_weights=[1.0]

droprate=0.25
n_filters = n_filters_start

conv1 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(inputs)
conv1 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

n_filters *= growth_factor
pool1 = BatchNormalization()(pool1)
conv2 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(pool1)
conv2 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
pool2 = Dropout(droprate)(pool2)

n_filters *= growth_factor
pool2 = BatchNormalization()(pool2)
conv3 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(pool2)
conv3 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv3)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
pool3 = Dropout(droprate)(pool3)

n_filters *= growth_factor
pool3 = BatchNormalization()(pool3)
conv4_0 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(pool3)
conv4_0 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv4_0)
pool4_1 = MaxPooling2D(pool_size=(2, 2))(conv4_0)
pool4_1 = Dropout(droprate)(pool4_1)

n_filters *= growth_factor
pool4_1 = BatchNormalization()(pool4_1)
conv4_1 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(pool4_1)
conv4_1 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv4_1)
pool4_2 = MaxPooling2D(pool_size=(2, 2))(conv4_1)
pool4_2 = Dropout(droprate)(pool4_2)

n_filters *= growth_factor
conv5 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(pool4_2)
conv5 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv5)

n_filters //= growth_factor
if upconv:
    up6_1 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv5), conv4_1])
else:
    up6_1 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4_1])
up6_1 = BatchNormalization()(up6_1)
conv6_1 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(up6_1)
conv6_1 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv6_1)
conv6_1 = Dropout(droprate)(conv6_1)

n_filters //= growth_factor
if upconv:
    up6_2 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv6_1), conv4_0])
else:
    up6_2 = concatenate([UpSampling2D(size=(2, 2))(conv6_1), conv4_0])
up6_2 = BatchNormalization()(up6_2)
conv6_2 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(up6_2)
conv6_2 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv6_2)
conv6_2 = Dropout(droprate)(conv6_2)

n_filters //= growth_factor
if upconv:
    up7 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv6_2), conv3])
else:
    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6_2), conv3])
up7 = BatchNormalization()(up7)
conv7 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(up7)
conv7 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv7)
conv7 = Dropout(droprate)(conv7)

n_filters //= growth_factor
if upconv:
    up8 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv7), conv2])
else:
    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2])
up8 = BatchNormalization()(up8)
conv8 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(up8)
conv8 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv8)
conv8 = Dropout(droprate)(conv8)

n_filters //= growth_factor
if upconv:
    up9 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv8), conv1])
else:
    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1])
conv9 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(up9)
conv9 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv9)

conv10 = Conv2D(n_classes, (1, 1), activation='sigmoid')(conv9)



import tensorflow as tf
from tensorflow import keras
image_size = 256
def bn_act(x, act=True):
    x = keras.layers.BatchNormalization()(x)
    if act == True:
        x = keras.layers.Activation("relu")(x)
    return x

def conv_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    conv = bn_act(x)
    conv = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(conv)
    return conv

def stem(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    conv = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
    conv = conv_block(conv, filters, kernel_size=kernel_size, padding=padding, strides=strides)

    shortcut = keras.layers.Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
    shortcut = bn_act(shortcut, act=False)

    output = keras.layers.Add()([conv, shortcut])
    return output

def residual_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    res = conv_block(x, filters, kernel_size=kernel_size, padding=padding, strides=strides)
    res = conv_block(res, filters, kernel_size=kernel_size, padding=padding, strides=1)

    shortcut = keras.layers.Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
    shortcut = bn_act(shortcut, act=False)

    output = keras.layers.Add()([shortcut, res])
    return output

def upsample_concat_block(x, xskip):
    u = keras.layers.UpSampling2D((2, 2))(x)
    c = keras.layers.Concatenate()([u, xskip])
    return c

def ResUNet():
    f = [16, 32, 64, 128, 256]
    inputs = keras.layers.Input((image_size, image_size, 3))

    ## Encoder
    e0 = inputs
    e1 = stem(e0, f[0])
    e2 = residual_block(e1, f[1], strides=2)
    e3 = residual_block(e2, f[2], strides=2)
    e4 = residual_block(e3, f[3], strides=2)
    e5 = residual_block(e4, f[4], strides=2)

    ## Bridge
    b0 = conv_block(e5, f[4], strides=1)
    b1 = conv_block(b0, f[4], strides=1)

    ## Decoder
    u1 = upsample_concat_block(b1, e4)
    d1 = residual_block(u1, f[4])

    u2 = upsample_concat_block(d1, e3)
    d2 = residual_block(u2, f[3])

    u3 = upsample_concat_block(d2, e2)
    d3 = residual_block(u3, f[2])

    u4 = upsample_concat_block(d3, e1)
    d4 = residual_block(u4, f[1])

    outputs = keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(d4)
    model = keras.models.Model(inputs, outputs)
    return model

model = ResUNet()
print(model)

model.compile(
    optimizer=Adam(lr=init_learning_rate, decay=init_learning_rate / Num_Epochs),
    loss=weighted_binary_crossentropy,
    metrics=['accuracy', iou, dice_coef, jaccard_coef])

model.summary()

from tqdm.keras import TqdmCallback

callbacks = [
            ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=10, min_lr=1e-9, epsilon=0.00001, verbose=1, mode='min'),]

history = model.fit(x_train, y_train, epochs=50, validation_data = (x_test, y_test), batch_size=BatchSize, verbose=0,shuffle=True,steps_per_epoch=None, callbacks=[TqdmCallback(verbose=1)])




model.save("resunetseg.h5")
