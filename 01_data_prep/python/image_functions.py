import os
from os import walk
import cv2
import numpy as np
import random

def resize_short_axis_to_pixel(image: np.ndarray, short_side_pixels: int) -> np.ndarray:
    original_width: int = image.shape[1]
    original_height: int = image.shape[0]

    if original_width < original_height:
        new_width: int = int(short_side_pixels)
        new_height: int = int(original_height * new_width / original_width)
    elif original_height < original_width:
        new_height: int = int(short_side_pixels)
        new_width: int = int(original_width * new_height / original_height) 
    else:
        new_height: int = int(short_side_pixels)
        new_width: int = int(short_side_pixels)

    image = cv2.resize(image, (new_width, new_height), interpolation = cv2.INTER_AREA)

    return image

def get_short_side_sized_squared_center_cut(image: np.ndarray) -> np.ndarray:
    original_width: int = image.shape[1]
    original_height: int = image.shape[0]

    if original_width < original_height:
        size = original_width
        height_center = original_height / 2
        targeted_center = original_width / 2
        diff_on_each_side = height_center - targeted_center
        height_start = int(diff_on_each_side)
        height_end = int(original_height - diff_on_each_side)
        image = image[height_start:height_end, :]
    else:
        size = original_height
        width_center = original_width / 2
        targeted_center = original_height / 2
        diff_on_each_side = width_center - targeted_center
        width_start = int(diff_on_each_side)
        width_end = int(original_width - diff_on_each_side)
        image = image[:, width_start:width_end]

    #image = cv2.resize(image, (size, size), interpolation = cv2.INTER_AREA)
    return image

def detectBird(image: np.ndarray):
    birdsCascade = cv2.CascadeClassifier("../ext/birds.xml")
    # convert the frame into gray scale for better analysis
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect birds in the gray scale image
    birds = birdsCascade.detectMultiScale(
        image,
        scaleFactor=1.4,
        minNeighbors=5,
        minSize=(10, 10),
        maxSize=(448, 448),
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the detected birds approaching the farm
    for (x, y, w, h) in birds:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 200, 0), 2)

    # Display the resulting frame
    cv2.imshow('Birds', image)
    cv2.waitKey(500)
    cv2.destroyAllWindows()

def generate_standard_squares(input_dir, output_dir):
    for (dirpath, dirnames, filenames) in walk(input_dir):
        if len(filenames) != 0:
            dirname = os.path.basename(dirpath)
            output_path = os.path.join(output_dir, dirname)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            for filename in filenames:
                image: np.ndarray = cv2.imread(os.path.join(dirpath, filename))
                image: np.ndarray = resize_short_axis_to_pixel(image, 224)
                image: np.ndarray = get_short_side_sized_squared_center_cut(image)
                cv2.imwrite(os.path.join(output_path, filename), image)

def generate_horizontal_flips(input_dir, output_dir):
    for (dirpath, dirnames, filenames) in walk(input_dir):
        if len(filenames) != 0:
            dirname = os.path.basename(dirpath)
            output_path = os.path.join(output_dir, dirname)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            for filename in filenames:
                image: np.ndarray = cv2.imread(os.path.join(dirpath, filename))
                image: np.ndarray = cv2.flip(image, 1)
                cv2.imwrite(os.path.join(output_path, filename), image)

def generate_randomized_brightness_and_contrast_images(input_dir, output_dir):
    for (dirpath, dirnames, filenames) in walk(input_dir):
        if len(filenames) != 0:
            dirname = os.path.basename(dirpath)
            output_path = os.path.join(output_dir, dirname)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            for filename in filenames:
                image: np.ndarray = cv2.imread(os.path.join(dirpath, filename))
                new_image = image
                alpha = random.uniform(0.7, 3.0)
                beta = random.randrange(0, 101)
                new_image = alpha*image+beta
                cv2.imwrite(os.path.join(output_path, filename), new_image)

def unite_directories(input_dirs: list, output_dir: str):
    for input_dir in input_dirs:
        for (dirpath, dirnames, filenames) in walk(input_dir):
            if len(filenames) != 0:
                dirname = os.path.basename(dirpath)
                output_path = os.path.join(output_dir, dirname)
                if not os.path.exists(output_path):
                    os.makedirs(output_path)

def test():
    path = '../../02_data/02_manual_deletion'
    for (dirpath, dirnames, filenames) in walk(path):
        if len(filenames) != 0:
            for filename in filenames:
                image: np.ndarray = cv2.imread(os.path.join(dirpath, filename))
                image = resize_short_axis_to_pixel(image, 224)
                image = get_short_side_sized_squared_center_cut(image)
                cv2.imshow('Birds', image)
                cv2.waitKey(500)
                cv2.destroyAllWindows()