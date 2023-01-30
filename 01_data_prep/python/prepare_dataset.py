import os
from sklearn.model_selection import train_test_split
from os import walk, path
import random
import shutil

def create_train_test_split(workdir: str, input_directory: str, output_directory: str):
    input_path = path.join(workdir, input_directory)
    output_path = path.join(workdir, output_directory)

    image_list = []

    for (dirpath, dirnames, filenames) in walk(input_path):
        label = path.basename(dirpath)
        
        for filename in filenames:
            
            filepath = path.join(dirpath, filename)
            image_list.append([label, filepath])
        
    labels = [i[0] for i in image_list]
    image_list_train, image_list_test = train_test_split(image_list, test_size=0.2, random_state=42, stratify=labels)
    print(len(image_list_train))
    print(len(image_list_test))
    print(len([amsel_train for amsel_train in image_list_train if amsel_train[0] == 'amsel']))
    print(len([amsel_test for amsel_test in image_list_test if amsel_test[0] == 'amsel']))

    i=1 #just to ensure different names for all images (because of the copies)

    for train_item in image_list_train:
        filename = f'{i}.jpg'
        subpath = path.join(output_path, 'train_images', train_item[0])
        if not os.path.exists(subpath):
            os.makedirs(subpath)
        shutil.copy(train_item[1], path.join(subpath, filename))
        i=i+1
    for test_item in image_list_test:
        filename = f'{i}.jpg'
        subpath = path.join(output_path, 'test_images', test_item[0])
        if not os.path.exists(subpath):
            os.makedirs(subpath)
        shutil.copy(test_item[1], path.join(subpath, filename))
        i=i+1

def prepare_dataset(workdir: str, input_directories: list, output_directory_name: str):
    image_list = []

    for directory in input_directories:
        cur_path = path.join(workdir, directory)
        #print(cur_path)
        for directory_2 in ['train_images', 'test_images']:
            cur_tt_path = path.join(cur_path, directory_2)
            for (dirpath, dirnames, filenames) in walk(cur_tt_path):
                label = path.basename(dirpath)
                for filname in filenames:
                    filepath = path.join(cur_tt_path, label, filname)
                    image_list.append([label, directory_2, filepath])

    output_dir = path.join(workdir, output_directory_name)

    for item in image_list:
        subpath = path.join(output_dir, item[1], item[0])
        if not os.path.exists(subpath):
            os.makedirs(subpath)
        shutil.copy(item[2], path.join(subpath, path.basename(item[2])))
        
   #output_dir_zip = path.join(workdir, output_directory_name)
   #zip_file_path = path.join(output_dir_zip, dataset_name)
   #shutil.make_archive(zip_file_path, 'zip', output_dir)
