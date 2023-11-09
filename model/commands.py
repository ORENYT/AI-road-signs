import os

import keras.saving
import numpy as np
from keras.src.utils import load_img, img_to_array
from tensorflow.python.keras.saving.save import load_model

from model.image_generator import generate_images
from model.images_formatter import analyse_folders
from model.model import learn


def create_model():
    data = analyse_folders()
    print("Data collected")
    data, data_x, data_y = generate_images(data, 5, True)
    print(f"{len(data_x)} images were generated.")
    model = learn(data_x, data_y)
    model.save("signs_model")


def predict():
    classes = ["main_road", "no_way", "parking", "pedestrian_crossing"]

    model = keras.saving.load_model("signs_model")

    image_directory = "images"

    image_files = [
        f
        for f in os.listdir(image_directory)
        if f.endswith(".jpg") or f.endswith(".png")
    ]

    loaded_images = []

    for file in image_files:
        img = load_img(
            os.path.join(image_directory, file), target_size=(256, 256)
        )
        img_array = img_to_array(img)
        loaded_images.append(img_array)

    images_for_prediction = np.array(loaded_images)

    predictions = model.predict(images_for_prediction)

    for i, prediction in enumerate(predictions):
        predicted_class = classes[np.argmax(prediction)]
        print(f"Image: {image_files[i]} - Prediction: {predicted_class}")
