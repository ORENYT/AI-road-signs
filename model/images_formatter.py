from PIL import Image
import os
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
import numpy as np
from numpy import ndarray

IMAGES_COUNT = 15


def analyse_folders() -> dict:
    images_base = {
        "main_road": [],
        "no_way": [],
        "parking": [],
        "pedestrian_crossing": [],
    }
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.dirname(current_dir)
    database_dir = os.path.join(model_dir, "database")
    for element in images_base.keys():
        working_dir = os.path.join(database_dir, element)
        data = []
        for i in range(1, IMAGES_COUNT + 1):
            data.append(analyse_image(i, working_dir))
        images_base[element] = data
    return images_base


def analyse_image(i: int, input_dir: str) -> ndarray:
    file_name = f"{i}.jpg"
    img_path = os.path.join(input_dir, file_name)
    image = Image.open(img_path)
    image = image.resize((256, 256))
    image_array = np.array(image)
    return image_array


