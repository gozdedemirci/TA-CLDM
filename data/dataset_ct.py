from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import pandas as pd 
import numpy as np
import torch
import os

class XRayDataset(Dataset):
    
    def __init__(self, file_path='/home/gdemi/multi_view/ctspine1k', split="train", img_size=256):
        
        with open(os.path.join(file_path, split+".txt"), 'r') as f:
            lines = f.readlines()
        self.file_dir = [line.strip() for line in lines]
        self.file_path = file_path

        self.cls2idx = {'frontal': 0, 'lateral': 1}
        # print(f"Number of samples: {len(self.data_path)}")

        if split == "train":
            # Aggressive Augmentation for Small Datasets
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                # transforms.RandomHorizontalFlip(p=0.5),
                # transforms.RandomVerticalFlip(p=0.5),
                # transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.file_dir)
    
    def __getitem__(self, index):
        '''
        Returns a tuple of images and labels in order of: od_center_id, nasal_id, temporal_id
        '''
        patient_name = self.file_dir[index]
        cond_img = Image.open(os.path.join(self.file_path,"processed_data",patient_name,"img_lateral.png")).convert("L")
        fov_img = Image.open(os.path.join(self.file_path,"processed_data",patient_name,"img_frontal.png")).convert("L")

        fov_text = f'given lateral x-ray view, target is frontal x-ray view'

        # Apply identical augmentation to OD and FOV pairs
        seed = np.random.randint(2147483647)
        torch.manual_seed(seed)
        cond_img = self.transform(cond_img)
        torch.manual_seed(seed)  # Ensure same transform for FOV image
        fov_img = self.transform(fov_img)
        
        return {
            'cond_image': cond_img, # conditioned image
            'fov_image': fov_img, # input to the model
            'fov_text': fov_text,
            'patient': patient_name
            }

class XRayDatasetAE(Dataset):
    
    def __init__(self, file_path='/home/gdemi/multi_view/ctspine1k', split="train", img_size=256):
        
        with open(os.path.join(file_path, split+".txt"), 'r') as f:
            lines = f.readlines()
        self.file_dir = [line.strip() for line in lines]

        frontal = [(f, "frontal") for f in self.file_dir]
        lateral = [(f, "lateral") for f in self.file_dir]

        self.images = frontal + lateral

        self.file_path = file_path

        self.cls2idx = {'frontal': 0, 'lateral': 1}
        # print(f"Number of samples: {len(self.data_path)}")

        if split == "train":
            # Aggressive Augmentation for Small Datasets
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                # transforms.RandomHorizontalFlip(p=0.5),
                # transforms.RandomVerticalFlip(p=0.5),
                # transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        '''
        Returns a tuple of images and labels in order of: od_center_id, nasal_id, temporal_id
        '''
        patient_name, view = self.images[index]
        img = Image.open(os.path.join(self.file_path,"processed_data",patient_name,f"img_{view}.png")).convert("L")


        # Apply identical augmentation to OD and FOV pairs
        seed = np.random.randint(2147483647)
        torch.manual_seed(seed)
        img = self.transform(img)
        
        return {
            'source': img, # conditioned image
            'label': self.cls2idx[view], # input to the model
            }



if __name__ == "__main__":
    from torchvision import transforms
    import tqdm

    img_paths =  "/home/gdemi/multi_view/ctspine1k"

    train_ds = XRayDatasetAE('/home/gdemi/multi_view/ctspine1k', split="val", img_size=256)
    print('Train dataset', len(train_ds), '\n')
    # train_dl = torch.utils.data.DataLoader(train_ds, batch_size=3, shuffle=True)
    # for sample in train_dl:
    #     # print(sample['cond_image'][0].shape)
    #     # print(sample['fov_image'][0].shape)
    #     # print(sample['fov_text'][0])

    #     print(sample['source'][0].shape)
    #     print(sample['label'][0])
    #     break

    # train_ds = ROPDatasetAE('/home/gdemi/multi_view/multi_view_data', split="train", img_size=256)
    # train_dl = torch.utils.data.DataLoader(train_ds, batch_size=3, shuffle=True)
    # for sample in train_dl:
    #     print(sample['source'][0].shape)
    #     break