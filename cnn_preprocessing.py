import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import itertools
import os
import shutil
import random
import glob
import matplotlib.pyplot as plt
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
# pip install cv
import cv2
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

set_splits = ['train', 'validation', 'test']
TRAIN_PERC, VALID_PERC, TEST_PERC = 70, 20, 10

data_matrix = pd.read_csv('hubmap-organ-segmentation/train.csv')

data_matrix['path'] = data_matrix.apply(lambda row: 'hubmap-organ-segmentation/train_images/' +
                                                    str(row['id']) + '.tiff', axis=1)

labels = data_matrix['organ'].unique()


def attach_numerical(df):
    df['sex'].replace([0, 1], ['Female', 'Male'], inplace=True)
    df_relevant = df.filter(['id', 'age', 'sex', 'path', 'organ'], axis=1)
    npz_paths = []
    for i, row in df.iterrows():
        picture_path = row['path']
        npz_path = picture_path.split('.')[0] + '.npz'
        npz_paths.append(npz_path)

        pic_bgr_arr = cv2.imread(picture_path)
        pic_rgb_arr = cv2.cvtColor(pic_bgr_arr, cv2.COLOR_BGR2RGB)

        agee, sexx = row['age'], row['sex']

        stats = np.array([agee, sexx])

        organ = row['organ']

        np.savez_compressed(npz_path, pic=pic_rgb_arr, stats=stats, organ=organ)

    df_relevant['NPZ_Path'] = pd.Series(npz_paths)

    return df_relevant


# load in .npz files
# df1 = attach_numerical(data_matrix)


# if npz files are already loaded
def attach_already_created(df):
    df['sex'].replace([0, 1], ['Female', 'Male'], inplace=True)
    df_relevant = df.filter(['id', 'age', 'sex', 'path', 'organ'], axis=1)
    df_relevant['NPZ_Path'] = df_relevant.apply(lambda row: 'hubmap-organ-segmentation/train_images/' +
                                                            str(row['id']) + '.npz', axis=1)
    return df_relevant


df1 = attach_already_created(data_matrix)

print(df1)


# attach_numerical(data_matrix)


def create_scalers(df):
    df_stats = df[['age']]
    means = [df_stats[col].mean() for col in df_stats]
    std_devs = [df_stats[col].std() for col in df_stats]
    return means, std_devs


num_data_means, num_data_std_devs = create_scalers(df1)
num_data_means = np.append(num_data_means, [1])
num_data_std_devs = np.append(num_data_std_devs, [1])
print('means: ', num_data_means, 'std devs: ', num_data_std_devs)


def stat_scaler(tensor):
    return (tensor - num_data_means) / num_data_std_devs


plt.imshow(np.load('hubmap-organ-segmentation/train_images/62.npz')['pic'])

print(df1.head)

df1.drop(['age', 'sex'], inplace=True, axis=1)

shuffled_df = df1.sample(frac=1)

train_df, valid_df, test_df = shuffled_df[:int(len(shuffled_df) * 0.7)], \
                              shuffled_df[int(len(shuffled_df) * 0.7):int(len(shuffled_df) * 0.9)], \
                              shuffled_df[int(len(shuffled_df) * 0.9):]

print(len(train_df), len(valid_df), len(test_df))

print(shuffled_df.head)


def get_X_y(df):
    X_pic, X_stats = np.array([]), np.array([])
    y = np.array([])

    for name in df['NPZ_Path']:
        print(name)
        loaded_npz = np.load(name)

        pic = loaded_npz['pic']
        X_pic = np.append(X_pic, pic)

        stats = loaded_npz['stats']
        X_stats = np.append(X_stats, pic)

        y = np.append(y, loaded_npz['organ'])

    X_pic, X_stats = np.array(X_pic), np.array(X_stats)
    y = np.array(y)

    return (X_pic, X_stats), y


(X_train_pic, X_train_stats), y_train = get_X_y(train_df)
(X_valid_pic, X_valid_stats), y_valid = get_X_y(valid_df)
(X_test_pic, X_test_stats), y_test = get_X_y(test_df)

print((X_train_pic.shape, X_train_stats.shape), y_train.shape)

# define the picture (cnn) stream

input_pic = layers.Input(shape=(1000, 1000, 3))
x = layers.Lambda(preprocess_input)(input_pic)
x = MobileNetV2(input_shape=(1000, 1000, 3), include_top=False)(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(10, activation='relu')(x)
x = Model(inputs=input_pic, outputs=x)

# define the stats (feed forward) stream
input_stats = layers.Input(shape=(2,))
y = layers.Lambda(stat_scaler)(input_stats)
y = layers.Dense(32, activation='relu')(y)
y = layers.Dense(10, activation='relu')(y)
y = Model(inputs=input_stats, outputs=y)

# concatenate these two layers
combined = layers.concatenate([x.output, y.output])

z = layers.Dense(4, activation='relu')(combined)
z = layers.Dense(1, activation='softmax')(z)

model = Model(inputs=[x.input, y.input], outputs=z)

print(model.summary)

optimizer = Adam(learning_rate=0.001)
model.compile(loss='mse', optimizer=optimizer, metrics=['categorical_crossentropy'])

cp = ModelCheckpoint('model/', save_best_only=True)
model.fit(x=[X_train_pic, X_train_stats], y=y_train, validation_data=([X_valid_pic, X_valid_stats], y_valid),
          epochs=10, callbacks=[cp])
