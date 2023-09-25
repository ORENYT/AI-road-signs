import os.path

from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
import numpy as np


def generate_images(data: dict, multiply: int = 2, show: bool = False) -> tuple:
    if multiply < 2:
        raise ValueError("Incorrect multiplier, minimal amount is 2")
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    data_x = []
    data_y = []
    for element in data.keys():
        new_list = []
        for image in data[element]:
            image = np.expand_dims(image, axis=0)
            images_generator = datagen.flow(image, batch_size=1)
            for _ in range(multiply):
                batch = images_generator.next()
                augmented_image = batch[0].astype(np.uint8)
                new_list.append(augmented_image)
        data[element] = new_list
        data_x.extend(new_list)
        data_y.extend([element for _ in range(len(new_list))])
    if show:
        delete_previous_image_output()
        save_dir = os.path.curdir
        save_dir = os.path.join(save_dir, "reworked_images")
        for i, generated_image in enumerate(data_x):
            img = Image.fromarray(generated_image)
            file_path = os.path.join(save_dir, f"generated_image_{i}.jpg")
            img.save(file_path)
    return data, data_x, data_y


def delete_previous_image_output() -> None:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.dirname(current_dir)
    model_folder_path = os.path.join(model_dir, "model")
    folder_path = os.path.join(model_folder_path, "reworked_images")

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if os.path.isfile(file_path) and (
                filename.endswith(".jpg") or filename.endswith(".png")):
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Error deleting file {filename}: {str(e)}")
