import math
import shutil
import os
import requests
import tarfile
import random
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm
from PIL import Image
from typing import List, Callable

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision import models


def download(root):
    """Download the dataset"""
    downloads_dir = os.path.join(root, "downloads")
    data_dir = os.path.join(root, "images")
    try:
        shutil.rmtree(root)
    except FileNotFoundError:
        pass
    finally:
        os.mkdir(root)
        os.mkdir(downloads_dir)
    for url in [
        "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar",
        "http://vision.stanford.edu/aditya86/ImageNetDogs/lists.tar",
    ]:
        get_response = requests.get(url, stream=True)
        file_name = os.path.join(downloads_dir, url.split("/")[-1])
        with open(file_name, "wb") as f:
            print(f"Downloading {url}")
            for chunk in get_response.iter_content(chunk_size=1024):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
    for file in [
        os.path.join(downloads_dir, "images.tar"),
        os.path.join(downloads_dir, "lists.tar"),
    ]:
        with tarfile.open(file) as f:
            print(f"Extracting {file}")
            f.extractall(downloads_dir)
    # Split images into train, validation, and test sets
    print("Splitting dataset")
    os.mkdir(data_dir)
    os.mkdir(os.path.join(data_dir, "train"))
    os.mkdir(os.path.join(data_dir, "validation"))
    train_list = [f[0][0] for f in loadmat(os.path.join(downloads_dir, "train_list.mat"))["file_list"]]
    # Shuffle the training images
    random.shuffle(train_list)
    for (i, file) in enumerate(train_list):
        if i < 200:
            # The first 200 training images get put into the validation directory
            target_dir = os.path.join(data_dir, "validation")
        else:
            # The rest go into the train directory
            target_dir = os.path.join(data_dir, "train")
        try:
            # Create the directory for the breed if it doesn't exist
            os.mkdir(os.path.join(target_dir, os.path.split(file)[0]))
        except FileExistsError:
            # The directory was already there
            pass
        finally:
            # Move the image
            shutil.move(os.path.join(downloads_dir, "Images", file), os.path.join(target_dir, file))
    # Move the test images
    os.mkdir(os.path.join(data_dir, "test"))
    test_list = loadmat(os.path.join(downloads_dir, "test_list.mat"))["file_list"]
    for file in test_list:
        if not os.path.isdir(os.path.join(data_dir, "test", os.path.split(file[0][0])[0])):
            # Create the directory for the breed if it doesn't exist
            os.mkdir(os.path.join(data_dir, "test", os.path.split(file[0][0])[0]))
        # Move the image
        shutil.move(os.path.join(downloads_dir, "Images", file[0][0]), os.path.join(data_dir, "test", file[0][0]))
    print("Splitting complete")


def preprocess(image):
    width, height = image.size
    if width > height and width > 512:
        height = math.floor(512 * height / width)
        width = 512
    elif width < height and height > 512:
        width = math.floor(512 * width / height)
        height = 512
    pad_values = (
        (512 - width) // 2 + (0 if width % 2 == 0 else 1),
        (512 - height) // 2 + (0 if height % 2 == 0 else 1),
        (512 - width) // 2,
        (512 - height) // 2,
    )
    return T.Compose([
        T.Resize((height, width)),
        T.Pad(pad_values),
        T.ToTensor(),
        T.Lambda(lambda x: x[:3]),  # Remove the alpha channel if it's there
    ])(image)


def get_labels(root=os.getcwd()):
    subdirs = set()
    labels = {}
    for subdir, _, _ in os.walk(os.path.join(root, "data/images/test")):
        if (label := os.path.split(subdir)[-1]) != "test":
            subdirs |= {label}
            labels[label] = len(subdirs) - 1
    return labels


def get_dataset(root=os.getcwd(), set_type="test"):
    file_paths = []
    labels = []
    label_names = get_labels()
    if not os.path.isdir(os.path.join(root, "data/images")):
        download()
    for dirpath, _, files in os.walk(os.path.join(root, "data/images", set_type)):
        for file in files:
            file_paths.append(os.path.join(dirpath, file))
            labels.append(label_names[os.path.split(dirpath)[-1]])
    return file_paths, labels


def load_images(file_paths):
    images = []
    for path in file_paths:
        image = Image.open(path)
        images.append(image)
    return images


def predict(images):
    
    predictions = []
    
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(len(labels)), ncols=80):
        # for inputs, labels in tqdm(test_loader, ncols=80):
            image = preprocess(images[i])
            image = torch.from_numpy(np.asarray(image))
            image = image.unsqueeze(dim=0)
            prediction = model(image.to(DEVICE))
            prediction = prediction.argmax(dim=1).item()
            predictions.append(prediction)
            
    return predictions



DEVICE = torch.device("cpu")
LOAD_FILE = "resnet50.pt"


model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 120)
model.to(DEVICE)
model.load_state_dict(torch.load(LOAD_FILE, map_location=DEVICE), strict=False)
model.eval()


file_paths, labels = get_dataset(set_type="test")
images = load_images(file_paths[::100])
labels = labels[::100]
predictions = predict(images)