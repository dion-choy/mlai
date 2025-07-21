import os
from PIL import Image

def crop_to_aspect_ratio(image_path, save_path):
    with Image.open(image_path) as img:
        width, height = img.size
        aspect_ratio = width / height

        if aspect_ratio >= 1:
            target_ratio = 4 / 3
        else:  # Vertical
            target_ratio = 3 / 4

        if aspect_ratio > target_ratio:
            new_width = int(target_ratio * height)
            left = (width - new_width) // 2
            box = (left, 0, left + new_width, height)
        else:
            new_height = int(width / target_ratio)
            top = (height - new_height) // 2
            box = (0, top, width, top + new_height)

        cropped = img.crop(box)
        cropped.save(save_path)

def make_landscape(image_path, save_path):
    with Image.open(image_path) as img:
        width, height = img.size

        if height > width:
            img = img.rotate(-90, expand=True)

        img.save(save_path)

if __name__ == "__main__":

    path_source = "./raw/"
    source_file_list = os.listdir(path_source)

    num = 0
    for source_file in source_file_list:
        print(source_file)

        crop_to_aspect_ratio(path_source+source_file, path_source+source_file)
        make_landscape(path_source+source_file, path_source+source_file)

        with Image.open(path_source+source_file) as img:
            img = img.rotate(180, expand=True)
            img.save(path_source+f"unknown_{num}.jpg")
        num += 1

