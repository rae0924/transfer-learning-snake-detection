import os 
import torch
from torch.utils import data
from torch.utils.data import Dataset
from torchvision import transforms
import cv2


class SnakeDataset(Dataset):
    
    def __init__(self, root_dir='data', len_snake=0, len_non_snake=0):
        self.root_dir = os.path.join(os.path.dirname(__file__), root_dir)
        self.snake_dir = os.path.join(self.root_dir, 'snake')
        self.non_snake_dir = os.path.join(self.root_dir, 'non-snake')
        for subdir in os.listdir(self.snake_dir):
            len_snake += len(os.listdir(os.path.join(self.snake_dir,subdir))) 
        for subdir in os.listdir(self.non_snake_dir):
            len_non_snake += len(os.listdir(os.path.join(self.non_snake_dir,subdir))) 
        self.len_snake = len_snake
        self.len_non_snake = len_non_snake
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])


    def __len__(self):
        return self.len_snake + self.len_non_snake

    def __getitem__(self, idx):
        cnt = 0
        for root,_,files in os.walk(self.root_dir):
            if len(files) != 0:
                if idx < (cnt + len(files)):
                    img_path = os.path.join(root, files[idx-cnt])
                    break
                cnt+=len(files)
        if img_path.find("non-snake") != -1:
            label = torch.tensor([0]).float()
        else:
            label = torch.tensor([1]).float()
        try:
            img = (cv2.imread(img_path)[:,:,::-1])
        except TypeError as e:
            print(f"Error loading image: {img_path}")
        img = cv2.resize(img, (224,224))
        img = torch.tensor(img, dtype=torch.float)
        img = img/255
        img = img.permute(2,0,1)
        img = self.normalize(img)
        sample = (img, label)
        return sample