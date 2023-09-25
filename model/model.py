from tensorflow import keras

from keras.models import Sequential
from keras.layers import (
    Dense,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dropout,
    BatchNormalization,
)
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, Adadelta
from keras import utils
from keras.preprocessing import image

from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

import random
import warnings
import math
import os


warnings.filterwarnings(
    "ignore", category=matplotlib.MatplotlibDeprecationWarning
)
classes = ["main_road", "no_way", "parking", "pedestrian_crossing"]


def learn(data_x: list, data_y: list):
    x_train, y_train, x_test, y_test = train_test_split(
        data_x, data_y, test_size=0.2, shuffle=True
    )
