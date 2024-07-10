import json
import os
import zipfile
from PIL import Image

def unzip_data(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def load_json(json_path):
    with open(json_path) as f:
        return json.load(f)

def load_images(image_dir):
    images = []
    for file_name in os.listdir(image_dir):
        if file_name.endswith('.jpg') or file_name.endswith('.png'):
            img = Image.open(os.path.join(image_dir, file_name))
            images.append((file_name, img))
    return images

def get_asset_distribution(concepts_data):
    asset_counts = {}
    for concept in concepts_data:
        for frame in concept['Implementation']:
            for asset in frame['Asset-Suggestions']:
                asset_type = asset['category']
                if asset_type not in asset_counts:
                    asset_counts[asset_type] = 0
                asset_counts[asset_type] += 1
    return asset_counts

def get_frame_statistics(concepts_data):
    num_frames = [len(concept['Implementation']) for concept in concepts_data]
    avg_frames_per_concept = sum(num_frames) / len(num_frames)
    return avg_frames_per_concept
