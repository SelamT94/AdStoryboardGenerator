import os
import Image

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_image(image, path):
    image.save(path)

def load_image(image_path):
    return Image.open(image_path)
