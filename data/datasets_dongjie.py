from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import pandas as pd 
import numpy as np
import random
import torch
import os

    
class ROPDatasetDongJie(Dataset):
    def __init__(self, file_path='/home/ROP_DATASET/fov_data_dongjie', split="train", img_size=256):
        self.file_path = file_path
        self.file_dir = pd.read_csv(os.path.join(file_path, split+".csv"))
        self.patients = self.file_dir[['patient_id', 'folder', 'side_eye']].values # we consider all patients and FOV's
        # print(f"Number of patients: {len(self.patients)}")

        self.side2word = {'OD': 'right', 'OS': 'left'}

        self.data_path = self.load_data_path()

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
    
    def load_data_path(self):
        self.data = []
        for patient in self.patients:
            files = self.file_dir[(self.file_dir['patient_id'] == patient[0]) & (self.file_dir['folder'] == patient[1]) & (self.file_dir['side_eye'] == patient[2])]
            files = files.iloc[0,:]
            fov_folders = os.listdir(os.path.join(self.file_path, files['path']))
            od_path, nasal_path, temporal_path = None, None, None
            # try to get all fovs: C, N, T and if not, get whatever we found
            if "OD_Center" in fov_folders:
                od_files = os.listdir(os.path.join(self.file_path, files['path'], "OD_Center"))
                # od_path = os.path.join(files['path'], "OD_Center", os.listdir(os.path.join(self.file_path, files['path'], "OD_Center"))[random.randint(0, len(od_files)-1)])
                od_path = os.path.join(files['path'], "OD_Center", os.listdir(os.path.join(self.file_path, files['path'], "OD_Center"))[0])
            if "Nasal" in fov_folders:
                nasal_files = os.listdir(os.path.join(self.file_path, files['path'], "Nasal"))
                nasal_path = os.path.join(files['path'], "Nasal", os.listdir(os.path.join(self.file_path, files['path'], "Nasal"))[0])
            if "Temporal" in fov_folders:
                temporal_files = os.listdir(os.path.join(self.file_path, files['path'], "Temporal"))
                temporal_path = os.path.join(files['path'], "Temporal", os.listdir(os.path.join(self.file_path, files['path'], "Temporal"))[0])

            # in data path we will use all possible combinations of two images
            if all((od_path, nasal_path, temporal_path)):
                self.data.append((files, patient, od_path, nasal_path, 'center2nasal'))
                self.data.append((files, patient, od_path, temporal_path, 'center2temporal'))
                # self.data.append((files, patient, nasal_path, temporal_path, 'nasal2temporal'))
                # self.data.append((files, patient, nasal_path, od_path, 'nasal2center'))
                # self.data.append((files, patient, temporal_path, od_path, 'temporal2center'))
                # self.data.append((files, patient, temporal_path, nasal_path, 'temporal2nasal'))
            elif all((od_path, nasal_path)):
                self.data.append((files, patient, od_path, nasal_path, 'center2nasal'))
                # self.data.append((files, patient, nasal_path, od_path, 'nasal2center'))
            elif all((od_path, temporal_path)): 
                self.data.append((files, patient, od_path, temporal_path, 'center2temporal'))
                # self.data.append((files, patient, temporal_path, od_path, 'temporal2center'))
            # elif all((nasal_path, temporal_path)):
            #     self.data.append((files, patient, nasal_path, temporal_path, 'nasal2temporal'))
            #     self.data.append((files, patient, temporal_path, nasal_path, 'temporal2nasal'))
            else:
                continue
        return self.data

    def __len__(self):
        return len(self.data_path)
    
    def __getitem__(self, index):
        '''
        Returns a tuple of images and labels in order of: od_center_id, nasal_id, temporal_id
        '''
        file_info, patient, fov_id1, fov_id2, fov = self.data_path[index]
        od_img = Image.open(os.path.join(self.file_path,fov_id1))
        fov_img = Image.open(os.path.join(self.file_path,fov_id2))

        cond, target = fov.split("2")
        _, _, _, side, zone_info, stage_info, plus_info, type_info = file_info

        base_text = f"A fundus image of a {self.side2word[side]} eye, generating a {target} view from a {cond} view."

        # Build clinical text by including only available info
        clinical_parts = []
        if zone_info != 9999.0:
            clinical_parts.append(f"ROP Zone {zone_info}")
        if stage_info != 9999.0:
            clinical_parts.append(f"Stage {stage_info}")
        if plus_info != 9999.0:
            clinical_parts.append(f"{'with' if plus_info=='1' else 'without'} Plus disease")
        if type_info != 9999.0:
            clinical_parts.append(f"Type {type_info}")
        if clinical_parts:
            clinical_text = " The patient has " + ", ".join(clinical_parts) + "."
        else:
            clinical_text = ""
        fov_text = base_text + clinical_text

        # Apply identical augmentation to OD and FOV pairs
        seed = np.random.randint(2147483647)
        torch.manual_seed(seed)
        od_img = self.transform(od_img)
        torch.manual_seed(seed)  # Ensure same transform for FOV image
        fov_img = self.transform(fov_img)
        
        return {
            'cond_image': od_img, # conditioned image
            'fov_image': fov_img, # input to the model
            'fov_text': fov_text,
            'patient': "_".join([str(x) for x in patient])
            }

    
class ROPDatasetDongJieAE(Dataset):
    
    def __init__(self, file_path='/home/ROP_DATASET/fov_data_dongjie', split="train", img_size=256):
        self.file_path = file_path
        self.file_dir = pd.read_csv(os.path.join(file_path, split+".csv"))
        self.patients = self.file_dir[['patient_id', 'folder', 'side_eye']].values # we consider all patients and FOV's
        # print(f"Number of patients: {len(self.patients)}")

        self.data_path = self.load_data_path()
        # print(f"Number of samples: {len(self.data_path)}")

        if split == "train":
            # Aggressive Augmentation for Small Datasets
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(15),
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
    
    def load_data_path(self):
        self.data = []
        for patient in self.patients:
            files = self.file_dir[(self.file_dir['patient_id'] == patient[0]) & (self.file_dir['folder'] == patient[1]) & (self.file_dir['side_eye'] == patient[2])]
            files = files.iloc[0,:]
            fov_folders = os.listdir(os.path.join(self.file_path, files['path']))
            od_path, nasal_path, temporal_path = None, None, None
            # try to get all fovs: C, N, T and if not, get whatever we found
            if "OD_Center" in fov_folders:
                od_files = os.listdir(os.path.join(self.file_path, files['path'], "OD_Center"))
                # od_path = os.path.join(files['path'], "OD_Center", os.listdir(os.path.join(self.file_path, files['path'], "OD_Center"))[random.randint(0, len(od_files)-1)])
                od_path = os.path.join(files['path'], "OD_Center", os.listdir(os.path.join(self.file_path, files['path'], "OD_Center"))[0])
            if "Nasal" in fov_folders:
                nasal_files = os.listdir(os.path.join(self.file_path, files['path'], "Nasal"))
                nasal_path = os.path.join(files['path'], "Nasal", os.listdir(os.path.join(self.file_path, files['path'], "Nasal"))[0])
            if "Temporal" in fov_folders:
                temporal_files = os.listdir(os.path.join(self.file_path, files['path'], "Temporal"))
                temporal_path = os.path.join(files['path'], "Temporal", os.listdir(os.path.join(self.file_path, files['path'], "Temporal"))[0])

            # in data path we will use all possible combinations of two images
            if all((od_path, nasal_path, temporal_path)):
                self.data.append((files, patient, od_path, nasal_path, 'center2nasal'))
                self.data.append((files, patient, od_path, temporal_path, 'center2temporal'))
                # self.data.append((files, patient, nasal_path, temporal_path, 'nasal2temporal'))
                # self.data.append((files, patient, nasal_path, od_path, 'nasal2center'))
                # self.data.append((files, patient, temporal_path, od_path, 'temporal2center'))
                # self.data.append((files, patient, temporal_path, nasal_path, 'temporal2nasal'))
            elif all((od_path, nasal_path)):
                self.data.append((files, patient, od_path, nasal_path, 'center2nasal'))
                # self.data.append((files, patient, nasal_path, od_path, 'nasal2center'))
            elif all((od_path, temporal_path)): 
                self.data.append((files, patient, od_path, temporal_path, 'center2temporal'))
                # self.data.append((files, patient, temporal_path, od_path, 'temporal2center'))
            # elif all((nasal_path, temporal_path)):
            #     self.data.append((files, patient, nasal_path, temporal_path, 'nasal2temporal'))
            #     self.data.append((files, patient, temporal_path, nasal_path, 'temporal2nasal'))
            else:
                continue
        return self.data

    def __len__(self):
        return len(self.data_path)
    
    def __getitem__(self, index):
        '''
        Returns a tuple of images and labels in order of: od_center_id, nasal_id, temporal_id
        '''
        file_info, patient, fov_id1, fov_id2, fov = self.data_path[index]
        _, _, _, side, zone_info, stage_info, plus_info, type_info = file_info

        img = Image.open(os.path.join(self.file_path,fov_id1))
        img = self.transform(img)
        
        return {
            'source': img, # conditioned image
            'label': type_info, # input to the model
            }


if __name__ == "__main__":
    from torchvision import transforms
    import tqdm
    import matplotlib.pyplot as plt

    img_paths =  "/home/ROP_DATASET/fov_data_dongjie"

    train_ds = ROPDatasetDongJie(img_paths, split="train", img_size=512)
    import pdb; pdb.set_trace()
    print('Train dataset', len(train_ds), '\n')
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=3, shuffle=True)
    for sample in train_dl:
        cond = sample['cond_image'][0]
        fov = sample['fov_image'][0]
        fov_text = sample['fov_text'][0]

        cond = (cond - cond.min()) / (cond.max() - cond.min())
        fov = (fov - fov.min()) / (fov.max() - fov.min())

        plt.figure(figsize=(14,4))
        plt.subplot(1,2,1)
        plt.imshow(np.transpose(cond.numpy(), (1, 2, 0)))
        plt.title('Conditioned Image')
        plt.axis('off')
        plt.subplot(1,2,2)
        plt.imshow(np.transpose(fov.numpy(), (1, 2, 0)))
        plt.title('FOV Image')
        plt.axis('off')
        plt.suptitle(fov_text)
        plt.show()
        plt.savefig('sample.png')

        
        break

    # train_ds = ROPDatasetDongJieAE('/home/ROP_DATASET/fov_data_dongjie', split="train", img_size=512)
    # train_dl = torch.utils.data.DataLoader(train_ds, batch_size=3, shuffle=True)
    # for sample in train_dl:
    #     print(sample['source'][0].shape)
    #     break