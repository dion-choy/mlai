import os
import random
import shutil


targets = ["validate/", "test/", "train/"]

def split(img_class):
    img_path = img_class + "/"
    path_source = "./raw/" + img_path
    next_source = "./datasets/"
    source_file_list = os.listdir(path_source)

    size = len(source_file_list)
    numFiles = [0.1*size,0.1*size,0.8*size]

    for index, split in enumerate(targets):
        for _ in range(int(numFiles[index])):
            num = random.randint(0, len(source_file_list)-1)

            shutil.move(path_source+source_file_list[num], f"{next_source}{split}{img_path}{source_file_list[num]}")

            source_file_list = os.listdir(path_source)

split("cauliflower")
split("broccoli")
split("unknown")
