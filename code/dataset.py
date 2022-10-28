import glob
import pickle
from typing import Iterator, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision
from torch.utils.data.dataset import T_co, Dataset

import albumentations as A


def torch_standardize_image(image: torch.Tensor, epsilon: float = 0.0000001):
    channels, _, _ = image.shape
    im_mean = image.view(channels, -1).mean(1).view(channels, 1, 1)
    im_std = image.view(channels, -1).std(1).view(channels, 1, 1)
    return (image - im_mean) / (im_std + epsilon)


class AnomalyDetectionDataset(Dataset):

    def __getitem__(self, index) -> T_co:
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def __iter__(self) -> Iterator[T_co]:
        return iter(self.data)

    def __init__(self, data):
        self.data = data


class CustomAnomalyDetectionDataset(Dataset):
    def __init__(self,
                 root_dir: str,
                 image_shape: Tuple[int, int],
                 aug_flag=False,
                 noise_flag=False,
                 transform=None,
                 noise_path='./noise/',
                 noise_alpha: int = 60,
                 noise_p=0.5,
                 ):
        list_files = sorted(glob.glob(root_dir + "/*"))
        assert len(list_files) != 0, "Error in loading frames"
        self.frames = list_files
        self.aug_flag = aug_flag
        self.transform = transform
        self.noise_path = noise_path
        self.image_shape = image_shape
        self.noise_flag = noise_flag
        self.noise_alpha = noise_alpha
        self.noise_p = noise_p
        self.labels = pd.read_csv(f"{root_dir}/../metadata/frames_labels.csv")

        # get label from frame_id using labels dataframe
        self.labels = self.labels.set_index("frame_id")
        # print(self.labels)
        if noise_flag:
            self.available_noises = len(glob.glob(noise_path + "/*"))
        else:
            self.available_noises = 0

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        pt_image = torchvision.io.read_image(self.frames[idx]).numpy()
        image = np.transpose(pt_image, (1, 2, 0))
        # we apply augmentations and apply standardization.
        if self.aug_flag and self.transform is not None:
            aug_image = self.transform(image=image)["image"]
            if self.noise_flag and np.random.rand() <= self.noise_p:
                int_image = self.add_noise_to_image(aug_image)
            else:
                int_image = aug_image
        else:
            int_image = image
        torch_float = torch.from_numpy(np.transpose(int_image,
                                                    (2, 0, 1))).float()
        # final_image = torch_float.div(255.0)
        final_image = torch_standardize_image(torch_float).contiguous()
        label = float(bool(self.labels.loc[idx]['label']))
        return final_image, label

    def add_noise_to_image(self, aug_image):
        noise_id = np.random.randint(self.available_noises)
        with open(self.noise_path + f"/noise_{noise_id}.pk", "rb") as pk_file:
            noise = pickle.load(pk_file)
        x_crop = np.random.randint(1000 - aug_image.shape[1])
        y_crop = np.random.randint(1000 - aug_image.shape[0])
        crop_noise = A.Crop(
            x_min=x_crop,
            y_min=y_crop,
            x_max=x_crop + aug_image.shape[1],
            y_max=y_crop + aug_image.shape[0],
            always_apply=True, p=1.0,
        )(image=noise)["image"]
        noise_rand_alpha = np.random.randint(
            low=self.noise_alpha, high=100) / 100.0
        final_image = aug_image / 255 + (crop_noise * (1.0 - noise_rand_alpha))
        return final_image
