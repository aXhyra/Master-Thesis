import copy
import json
from fractions import Fraction

import numpy as np
import torch
import albumentations as A
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from dataset.dataset import AnomalyDetectionDataset, \
    CustomAnomalyDetectionDataset
from utils import get_most_anomalous, get_most_informative


class DatasetHandler:
    def __init__(self, dataset, transformations, anomalous_class=(3,),
                 splits=(500, 500, 1000, 1000), shuffle=True,
                 torch_builtin=True,
                 use_all_classes_except_anomalous_as_normal=False,
                 split_size=50000, augment=False, noise_flag=False,
                 noise_path=None):
        """
        > The function takes in a dataset, a list of transformations, a tuple
        of splits, and a boolean value for shuffle. It then creates a
        train_set and a test_set, and splits the train_set into a train_set
        and a valid_set. It then creates a split_a, split_b, and split_c,
        and creates a dictionary of the splits

        :param dataset: the name of the dataset to use. This is the name of
        the dataset class in torchvision.datasets
        :param transformations: a
        list of transformations to apply to the data
        :param anomalous_class:
        the class of the anomalous data. In this case, it's the digit 3
        :param splits: tuple of 4 elements, first three elements define the
        dimension of A, B, and C splits, while the
        :param shuffle: Whether to
        shuffle the dataset or not, defaults to True (optional)
        :param torch_builtin: If True, the dataset is assumed to be a torch
        builtin  dataset. If False, the dataset is assumed to be a custom
        dataset, defaults to True (optional)
        :param  use_all_classes_except_anomalous_as_normal: This is a boolean
        parameter. If set to True, it will use all the classes except the
        anomalous class as the normal class. If set to False, it will use
        only the anomalous class as the normal class, defaults to False (
        optional)
        :param split_size: The number of images to be used in the split,
        defaults to 50000 (optional)
        """
        self.idx = None
        self.split_size = split_size
        self.shuffle = shuffle
        for i, split in enumerate(splits):
            if isinstance(split, Fraction):
                splits[i] = float(splits[i])

        if sum(splits) == 0:
            self.splits = None
        else:
            self.splits = splits

        if torch_builtin:
            train_set = datasets.__dict__[dataset](
                './data',
                train=True,
                download=True,
                transform=transforms.ToTensor())
            self.input_size = train_set.data[0].shape

            self.test_set = datasets.__dict__[dataset](
                './data',
                train=False,
                download=True,
                transform=transforms.ToTensor())

            train_set, valid_set = torch.utils.data.random_split(train_set, [
                int(len(train_set) * 0.95),
                int(len(train_set) * 0.05)])

            # Filtering the dataset to only contain the anomalous class.
            if use_all_classes_except_anomalous_as_normal:
                train_set = torch.utils.data.Subset(
                    train_set,
                    [i for i in range(len(train_set)) if
                     train_set[i][1] not in anomalous_class])

                self.valid_set = AnomalyDetectionDataset(
                    torch.utils.data.Subset(
                        valid_set,
                        [i for i in range(len(valid_set)) if
                         valid_set[i][1] not in anomalous_class]))
            else:
                train_set = torch.utils.data.Subset(
                    train_set,
                    [i for i in range(len(train_set)) if
                     train_set[i][1] in anomalous_class])

                self.valid_set = AnomalyDetectionDataset(
                    torch.utils.data.Subset(
                        valid_set,
                        [i for i in range(len(valid_set)) if
                         valid_set[i][1] in anomalous_class]))

        else:
            assert len(
                dataset) == 3, 'dataset must be a tuple of' \
                               '(train_set, test_set, valid_set) paths'
            train_set_path, test_set_path, valid_set_path = dataset

            if augment:
                image_size = (64, 64)

                composed_transform = A.Compose(
                    [
                        A.transforms.HorizontalFlip(p=0.5),
                        A.transforms.RandomBrightnessContrast(
                            brightness_limit=0.1, contrast_limit=0.1, p=0.5),
                        A.RandomSizedCrop(min_max_height=[50, image_size[0]],
                                          height=image_size[0],
                                          width=image_size[1], p=0.5),
                        A.Rotate(limit=10, p=0.5),
                    ]
                )
            else:
                composed_transform = transformations

            train_set = CustomAnomalyDetectionDataset(
                train_set_path, (64, 64),
                aug_flag=augment,
                noise_flag=noise_flag,
                transform=composed_transform,
                noise_path=noise_path)
            self.split_size = len(train_set)
            self.valid_set = CustomAnomalyDetectionDataset(
                valid_set_path,
                (64, 64),
                transform=transformations)
            self.test_set = CustomAnomalyDetectionDataset(
                test_set_path,
                (64, 64),
                transform=transformations)

        # Splitting the dataset into 3 parts, A, B, and C.
        self.data = train_set
        if self.splits is not None:
            if self.splits[0] < 1:
                self.splits[0] = int(len(train_set) * self.splits[0])
            if self.splits[1] < 1:
                self.splits[1] = int(len(train_set) * self.splits[1])
            if self.splits[2] < 1:
                self.splits[2] = int(len(train_set) * self.splits[2])

            split_a = self.splits[0]
            split_b = self.splits[1]

            self.splits[2] = self.splits[2] if len(self.splits) > 0 else len(
                train_set) - (split_b + split_a)

            self.train_part_a, self.train_part_b, self.train_part_c = \
                torch.utils.data.random_split(
                    train_set,
                    [split_a,
                     split_b,
                     len(train_set) -
                     (split_b +
                      split_a)])

            random_sample = np.random.choice(np.arange(len(self.train_part_c)),
                                             self.splits[2])
            random_sample = torch.utils.data.Subset(self.train_part_c,
                                                    random_sample)

            # save on json file the number of images per splits
            with open('./data/split_size.json', 'w') as fp:
                json.dump({'split_a': len(self.train_part_a),
                           'split_b': len(self.train_part_b),
                           'total_split_c': len(self.train_part_c)}, fp)

            augment_a = self.augment(torch.utils.data.ConcatDataset(
                [self.train_part_a, self.train_part_b]))
            augment_b = self.augment(
                torch.utils.data.ConcatDataset([self.train_part_a,
                                                self.train_part_b,
                                                random_sample]))

            self.split_a = AnomalyDetectionDataset(
                torch.utils.data.ConcatDataset([self.train_part_a,
                                                self.train_part_b,
                                                augment_a]),
            )
            self.split_b = AnomalyDetectionDataset(
                torch.utils.data.ConcatDataset([self.train_part_a,
                                                self.train_part_b,
                                                random_sample,
                                                augment_b]))
            self.split_c = None
            self.split_d = None
            self.split_e = None
            self.split_f = None
            self.split_g = None
            self.split_h = None
            self.split_i = None

            self.split_dict = {
                'AL1': self.split_a,
                'AL2': self.split_b,
                'AL3': self.split_c,
                'AL3.1': None,
                'AL3.2': self.split_d,
                'AL4': self.split_e,
                'AL5': self.split_f,
                'AL6': self.split_g,
                'AL7': self.split_h,
                'AL8': self.split_i,
                'DummyTrue': self.split_a,
                'DummyFalse': self.split_a,
                'DummyRandom': self.split_a
            }
        else:
            self.split_a = self.augment(train_set)
            self.split_b = None
            self.split_c = None
            self.split_d = None
            self.split_d_1 = None
            self.split_e = None
            self.split_f = None
            self.split_g = None
            self.split_h = None
            self.split_i = None
            self.split_dict = {
                'Upper': self.split_a,
                'AL1': self.split_a,
                'AL2': self.split_b,
                'AL3': self.split_c,
                'AL3.1': None,
                'AL3.2': self.split_d,
                'AL4': self.split_e,
                'AL5': self.split_f,
                'AL6': self.split_g,
                'AL7': self.split_h,
                'AL8': self.split_i,
                'DummyTrue': self.split_a,
                'DummyFalse': self.split_a,
                'DummyRandom': self.split_a
            }

    def augment(self, data):
        """
        This function takes a dataset and returns a new dataset with the same
        number of samples as the original dataset,
        but with the samples randomly selected from the original dataset

        :param data: the dataset to be augmented
        :return: a subset of the data.
        """
        if self.split_size - len(data) <= 0:
            return data

        idx = np.random.choice(np.arange(len(data)),
                               self.split_size - len(data))
        return torch.utils.data.Subset(data, idx)

    def gen_splits(self):
        """
        **gen_splits** should return a list of tuples of the form (train_set,
        test_set) where each tuple is a partition of
        the data
        """
        raise NotImplementedError("gen_splits is not yet implemented")

    def gen_split_c(self, model):
        """
        > We take the most anomalous data points from the train set,
        and augment them. Then we add them to the training set

        :param model: the model to use for anomaly detection
        """
        most_anomalous, _ = get_most_anomalous(model, self.train_part_c,
                                               self.splits[2])

        augment_c = self.augment(
            torch.utils.data.ConcatDataset([self.train_part_a,
                                            self.train_part_b,
                                            most_anomalous]))

        self.split_c = AnomalyDetectionDataset(
            torch.utils.data.ConcatDataset([self.train_part_a,
                                            self.train_part_b,
                                            most_anomalous,
                                            augment_c]))
        self.split_dict['AL3'] = self.split_c

    def gen_split_d(self, model):
        if self.split_d is None:
            most_anomalous, self.idx = get_most_anomalous(model,
                                                          self.train_part_c,
                                                          self.splits[3] // 2)
            self.split_d_1 = copy.deepcopy(most_anomalous)

            augment_d = self.augment(
                torch.utils.data.ConcatDataset([self.train_part_a,
                                                self.train_part_b,
                                                most_anomalous]))

            self.split_d = AnomalyDetectionDataset(
                torch.utils.data.ConcatDataset([self.train_part_a,
                                                self.train_part_b,
                                                most_anomalous,
                                                augment_d]))
            self.split_dict['AL3.1'] = copy.deepcopy(self.split_d)
        else:
            new_idx = [i for i in range(len(self.train_part_c)) if
                       i not in self.idx]
            part_c = torch.utils.data.Subset(self.train_part_c, new_idx)
            most_anomalous, _ = get_most_anomalous(model, part_c,
                                                   self.splits[3] // 2)
            augment_d = self.augment(
                torch.utils.data.ConcatDataset([self.train_part_a,
                                                self.train_part_b,
                                                self.split_d_1,
                                                most_anomalous]))
            self.split_d = AnomalyDetectionDataset(
                torch.utils.data.ConcatDataset([self.train_part_a,
                                                self.train_part_b,
                                                self.split_d_1,
                                                most_anomalous,
                                                augment_d]))
            self.split_dict['AL3.2'] = self.split_d

    def gen_split_e(self, model):
        most_informative = get_most_informative(model, self.train_part_c,
                                                torch.utils.data.ConcatDataset(
                                                    [self.train_part_a,
                                                     self.train_part_b]),
                                                self.splits[3],
                                                'latent_distance')

        augment_e = self.augment(
            torch.utils.data.ConcatDataset([self.train_part_a,
                                            self.train_part_b,
                                            most_informative]))

        self.split_e = AnomalyDetectionDataset(
            torch.utils.data.ConcatDataset([self.train_part_a,
                                            self.train_part_b,
                                            most_informative,
                                            augment_e]))
        self.split_dict['AL4'] = self.split_e

    def gen_split_f(self, model):
        most_informative = get_most_informative(model, self.train_part_c,
                                                self.split_a,
                                                self.splits[3] // 2,
                                                'latent_distance')
        most_informative2 = get_most_informative(model, self.train_part_c,
                                                 self.split_a,
                                                 self.splits[3] // 2 +
                                                 self.splits[3] % 2,
                                                 'error_map')

        most_informative = torch.utils.data.ConcatDataset(
            [most_informative, most_informative2])

        augment_f = self.augment(
            torch.utils.data.ConcatDataset([self.train_part_a,
                                            self.train_part_b,
                                            most_informative]))

        self.split_f = AnomalyDetectionDataset(
            torch.utils.data.ConcatDataset([self.train_part_a,
                                            self.train_part_b,
                                            most_informative,
                                            augment_f]))
        self.split_dict['AL5'] = self.split_f

    def gen_split_g(self, model):
        most_informative = get_most_informative(model, self.train_part_c,
                                                torch.utils.data.ConcatDataset(
                                                    [self.train_part_a,
                                                     self.train_part_b]),
                                                self.splits[3], 'minmax')

        augment_g = self.augment(
            torch.utils.data.ConcatDataset([self.train_part_a,
                                            self.train_part_b,
                                            most_informative]))

        self.split_g = AnomalyDetectionDataset(
            torch.utils.data.ConcatDataset([self.train_part_a,
                                            self.train_part_b,
                                            most_informative,
                                            augment_g]))
        self.split_dict['AL6'] = self.split_g

    def gen_split_h(self, model):
        most_informative = get_most_informative(model, self.train_part_c,
                                                torch.utils.data.ConcatDataset(
                                                    [self.train_part_a,
                                                     self.train_part_b]),
                                                self.splits[3],
                                                'minmax_iterative')

        augment_h = self.augment(
            torch.utils.data.ConcatDataset([self.train_part_a,
                                            self.train_part_b,
                                            most_informative]))

        self.split_h = AnomalyDetectionDataset(
            torch.utils.data.ConcatDataset([self.train_part_a,
                                            self.train_part_b,
                                            most_informative,
                                            augment_h]))
        self.split_dict['AL7'] = self.split_h

    def gen_split_h2(self, model):
        most_informative = get_most_informative(model, self.train_part_c,
                                                torch.utils.data.ConcatDataset(
                                                    [self.train_part_a,
                                                     self.train_part_b]),
                                                self.splits[3],
                                                'minmax_iterative2')

        augment_h = self.augment(
            torch.utils.data.ConcatDataset([self.train_part_a,
                                            self.train_part_b,
                                            most_informative]))

        self.split_h = AnomalyDetectionDataset(
            torch.utils.data.ConcatDataset([self.train_part_a,
                                            self.train_part_b,
                                            most_informative,
                                            augment_h]))
        self.split_dict['AL7'] = self.split_h

    def gen_split_i(self, model):
        most_informative = get_most_informative(model, self.train_part_c,
                                                torch.utils.data.ConcatDataset(
                                                    [self.train_part_a,
                                                     self.train_part_b]),
                                                self.splits[3],
                                                'minmax_anomaly')

        augment_i = self.augment(
            torch.utils.data.ConcatDataset([self.train_part_a,
                                            self.train_part_b,
                                            most_informative]))

        self.split_i = AnomalyDetectionDataset(
            torch.utils.data.ConcatDataset([self.train_part_a,
                                            self.train_part_b,
                                            most_informative,
                                            augment_i]))
        self.split_dict['AL8'] = self.split_i
