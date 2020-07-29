
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
import os
import cv2
import random
from skimage import io


class Data(Dataset):
    def __init__(self, root_dir, mode, device, crop_size=512, augment=True, caching=False):
        self.root_dir = root_dir
        self.mode = mode
        self.device = device
        self.crop_size = crop_size
        self.augment = augment
        self.caching = caching
        self.names = []
        self.tops = []
        self.heights = []
        self.load_names(root_dir, mode)

    def load_names(self, root_dir, mode):
        self.names = []
        self.tops = []
        self.heights = []
        file_path = os.path.join(root_dir, mode + '.txt')
        with open(file_path, "r") as f:
            for name in f:
                self.names.append(name.strip())
                self.tops.append(None)
                self.heights.append(None)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, item):
        image_top = self.tops[item]
        image_height = self.heights[item]
        if image_top is None:
            top_name = os.path.join(self.root_dir, 'top', self.names[item])
            # image_top = cv2.imread(top_name)
            # image_top = cv2.cvtColor(image_top, cv2.COLOR_BGR2RGB)
            image_top = io.imread(top_name)
            image_top = Image.fromarray(image_top)
            if self.caching:
                self.tops[item] = image_top

        if image_height is None:
            height_name = os.path.join(self.root_dir, 'dsm', self.names[item])
            image_height = io.imread(height_name)
            image_height = Image.fromarray(image_height)
            if self.caching:
                self.heights[item] = image_height

        self.crop_indices = transforms.RandomCrop.get_params(
            image_top, output_size=(self.crop_size, self.crop_size))
        i, j, h, w = self.crop_indices
        image_top = TF.crop(image_top, i, j, h, w)
        image_height = TF.crop(image_height, i, j, h, w)

        if self.mode == 'train' and self.augment and random.random() > 0.5:
            image_top = TF.hflip(image_top)
            image_height = TF.hflip(image_height)
        if self.mode == 'train' and self.augment and random.random() > 0.5:
            image_top = TF.vflip(image_top)
            image_height = TF.vflip(image_height)
        # if self.augment and random.random() > 0.5:
        #     image_top = TF.rotate(image_top, 90)
        #     image_height = TF.rotate(image_height, 90)

        image_top = TF.to_tensor(image_top)
        image_height = TF.to_tensor(image_height)

        return {'image': image_top.to(self.device),
                'height': image_height.to(self.device)}


