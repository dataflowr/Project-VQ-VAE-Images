import os
import shutil
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class StanfordCarsDataset(object):

    def __init__(self, batch_size, path, shuffle_dataset=True):
        if not os.path.isdir(path):
            os.mkdir(path)
            
        datasets.StanfordCars(
            root="../data/",
            split="train",
            download = True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
            ])
                )
        datasets.StanfordCars(
            root="../data/",
            split="test",
            download = True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
            ])
        )

        # create temporary directory structure with class folders
        temp_path_train = os.path.join(path, "temp_train")
        temp_path_test = os.path.join(path, "temp_test")
        
        if not os.path.isdir(temp_path_train):
            os.mkdir(temp_path_train)
            
        if not os.path.isdir(temp_path_test):
            os.mkdir(temp_path_test)
            
        for root, _, files in os.walk(os.path.join(path, "cars_train")):
            for file in files:
                src_path = os.path.join(root, file)
                class_folder = os.path.basename(root)
                dst_path = os.path.join(temp_path_train, class_folder, file)
                if not os.path.isdir(os.path.join(temp_path_train, class_folder)):
                    os.mkdir(os.path.join(temp_path_train, class_folder))
                shutil.copy(src_path, dst_path)

        self._training_data = datasets.ImageFolder(
            root=temp_path_train,
            transform=transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        )

        for root, _, files in os.walk(os.path.join(path, "cars_test")):
            for file in files:
                src_path = os.path.join(root, file)
                class_folder = os.path.basename(root)
                dst_path = os.path.join(temp_path_test, class_folder, file)
                if not os.path.isdir(os.path.join(temp_path_test, class_folder)):
                    os.mkdir(os.path.join(temp_path_test, class_folder))
                shutil.copy(src_path, dst_path)
                
        self._validation_data = datasets.ImageFolder(
            root=temp_path_test,
            transform=transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        )

        self._training_loader = DataLoader(
            self._training_data, 
            batch_size=batch_size, 
            num_workers=2,
            shuffle=shuffle_dataset,
            pin_memory=True
        )

        self._validation_loader = DataLoader(
            self._validation_data,
            batch_size=batch_size,
            num_workers=2,
            shuffle=shuffle_dataset,
            pin_memory=True
        )
        
        mean = torch.zeros(3)
        sq_mean = torch.zeros(3)
        for inputs, _ in self._training_loader:
            mean += inputs.mean(dim=(0, 2, 3))
            sq_mean += (inputs**2).mean(dim=(0, 2, 3))

        mean /= len(self._training_loader)
        sq_mean /= len(self._training_loader)
        self._train_data_variance = (sq_mean - mean**2).sum()

    @property
    def training_data(self):
        return self._training_data

    @property
    def validation_data(self):
        return self._validation_data

    @property
    def training_loader(self):
        return self._training_loader

    @property
    def validation_loader(self):
        return self._validation_loader

    @property
    def train_data_variance(self):
        return self._train_data_variance
