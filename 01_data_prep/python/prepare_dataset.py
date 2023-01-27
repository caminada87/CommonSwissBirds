import os
from sklearn.model_selection import train_test_split
from os import walk, path
import random
import shutil

def prepare_dataset(input_directories: list, output_directory_name: str, dataset_name: str):
    workdir = '..\\..\\02_data\\'
    image_list = []

    for directory in input_directories:
        cur_path = path.join(workdir, directory)
        #print(cur_path)
        for (dirpath, dirnames, filenames) in walk(cur_path):
            label = path.basename(dirpath)
            for filname in filenames:
                filepath = path.join(cur_path, label, filname)
                image_list.append([label, filepath])

    random.shuffle(image_list)
    labels = [i[0] for i in image_list]

    train, test = train_test_split(image_list, test_size=0.2, random_state=0, stratify=labels)

    print(len(train))
    print(len(test))

    output_dir = path.join(workdir, output_directory_name, dataset_name)
    output_dir_train = path.join(output_dir, 'train_images')
    output_dir_test = path.join(output_dir, 'test_images')

    i=1 #just to ensure different names for all images (because of the copies)
    for train_item in train:
        filename = f'{i}.jpg'
        subpath = path.join(output_dir_train, train_item[0])
        if not os.path.exists(subpath):
            os.makedirs(subpath)
        shutil.copy(train_item[1], path.join(subpath, filename))
        i=i+1
    for test_item in test:
        filename = f'{i}.jpg'
        subpath = path.join(output_dir_test, test_item[0])
        if not os.path.exists(subpath):
            os.makedirs(subpath)
        shutil.copy(test_item[1], path.join(subpath, filename))
        i=i+1
        
    output_dir_zip = path.join(workdir, output_directory_name)
    zip_file_path = path.join(output_dir_zip, dataset_name)
    shutil.make_archive(zip_file_path, 'zip', output_dir)


input_directories: list = ['03_standard_squares',
    '04_horizontal_flips',
    '05_randomized_brightness_contrast',
    '06_flipped_randomized_brightness_contrast'
]
output_directory_name: str = '98_dataset_preparation'
dataset_name: str = 'common_swiss_birds'

prepare_dataset(input_directories, output_directory_name, dataset_name)