from images_formatter import analyse_folders, IMAGES_COUNT
from image_generator import generate_images


def main() -> None:
    data = analyse_folders()
    print("Data collected")
    data, data_x, data_y = generate_images(data, 2, True)
    print(f"{len(data_x)} images were generated.")


if __name__ == "__main__":
    main()
