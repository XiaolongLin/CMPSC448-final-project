import torch
import torch.nn as nn
from torch.utils.data import Dataset
from pathlib import Path
import os
import torchvision
from PIL import Image
import pandas as pd


class AnimalDataset(Dataset):
    def __init__(self, pair_file, data_fold, transformers=None) -> None:
        super().__init__()
        df = pd.read_csv(pair_file)
        self.img_name_list = df['img_name'].values
        self.label_list = df['label'].values
        self.data_fold = Path(data_fold)
        
        if transformers == None:
            self.transformers = torchvision.transforms.Compose([
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
        else:
            self.transformers = transformers
    
    
    def __getitem__(self, index):
        img_path, label = self.data_fold / self.img_name_list[index], self.label_list[index]
        img = Image.open(img_path)
        img = self.transformers(img)
        return img, label
                
                
    def __len__(self):
        return len(self.label_list)
    