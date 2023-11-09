from keras.src.utils import to_categorical

from keras.models import Sequential
from keras.layers import (
    Dense,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dropout,
    BatchNormalization,
)

from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import warnings

from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings(
    "ignore", category=matplotlib.MatplotlibDeprecationWarning
)
classes = ["main_road", "no_way", "parking", "pedestrian_crossing"]


def learn(data_x: list, data_y: list):
    label_encoder = LabelEncoder()
    data_y = np.array(data_y)
    data_y = label_encoder.fit_transform(data_y)
    data_y = to_categorical(data_y, num_classes=len(classes))

    data_x = np.array(data_x)
    x_train, x_test, y_train, y_test = train_test_split(
        data_x, data_y, test_size=0.2, shuffle=True
    )

    model = Sequential(
        [
            Conv2D(32, (3, 3), activation="relu", input_shape=(256, 256, 3)),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            Conv2D(64, (3, 3), activation="relu"),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            Flatten(),
            Dense(128, activation="relu"),
            Dropout(0.4),
            Dense(4, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    history = model.fit(
        x_train,
        y_train,
        epochs=10,
        validation_data=(x_test, y_test),
        batch_size=32,
    )

    plot_history(history)

    return model


def plot_history(history):
    plt.plot(history.history["accuracy"], label="Training accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation accuracy")
    plt.title("Model Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    epochs = range(1, len(history.history["accuracy"]) + 1)
    plt.xticks(epochs)
    plt.grid(True)
    plt.legend()
    plt.show()
