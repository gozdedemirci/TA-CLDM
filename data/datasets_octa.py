import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import pydicom

import sys
sys.path.append("/home/gdemi/multi_view/")

from src.models.attention_unet import AttentionUNet
import src.datasets.transforms as T

unet = AttentionUNet(n_channels=3, n_classes=1)
cp = torch.load("/home/gdemi/topo_uda/temp/anotha_final/results/OCTA5002Supervised-AttenUnet-20260121_032822/models/best_loss.pth")
# cp = torch.load("/home/gdemi/topo_uda/temp/logs/chase2rop.pth")
unet.load_state_dict(cp)
vessel_transformer = T.Compose([T.NormalizeAndTranspose()])

class OCTADataset(Dataset):
    def __init__(self, csv_file, root_dir, mode='train'):
        """
        Args:
            csv_file (string): Path to the csv file.
            root_dir (string): Path to the dataset root (e.g. '.../fov_data_dongjie').
        """
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.mode = mode
        
        self.patient_data = []
        self.__get_patient_data__()

        if mode == "train":
            # Aggressive Augmentation for Small Datasets
            self.transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.patient_data)


    def generate_prompts(self, patient, source_view, target_view):

        side = patient[-1] # Updated column name
        side = "right" if side == 'R' else "left"
        
        # Base Text
        base_text = f"Given {source_view} view, generate {target_view} view of {side} eye"

        return base_text

    def __get_patient_data__(self):
        # Define the target/source strategy
        target_candidates = 'Optic Disc, 6 x 6'
        source_view_name = 'Macula, 6 x 6'

        patients = self.data_frame.patient.unique()

        for patient in patients:
            views = self.data_frame[self.data_frame.patient == patient]

            source_path = views[views['anatomic_region'] == source_view_name]['associated_enface_1_file_path'].values[0]
            target_path = views[views['anatomic_region'] == target_candidates]['associated_enface_1_file_path'].values[0]
            
            self.patient_data.append({
                'source_path': source_path,
                'target_path': target_path,
                'condition_view_name': source_view_name,
                'target_view_name': target_candidates,
                'patient_info': patient
            })

    def __getitem__(self, idx):

        row = self.patient_data[idx]
        
        # Define the target/source strategy
        target_view_name = row['target_view_name']
        source_view_name = row['condition_view_name']

        source_path = row['source_path']
        target_path = row['target_path']

        # 3. Load Images
        source_img = pydicom.dcmread(os.path.join(self.root_dir, source_path)).pixel_array
        target_img = pydicom.dcmread(os.path.join(self.root_dir, target_path)).pixel_array

        source_img = Image.fromarray(source_img).convert('L')
        target_img = Image.fromarray(target_img).convert('L')

        # Apply identical augmentation to OD and FOV pairs
        seed = np.random.randint(2147483647)
        torch.manual_seed(seed)
        source_img = self.transform(source_img)
        torch.manual_seed(seed)  # Ensure same transform for FOV image
        target_img = self.transform(target_img)
            
        # 4. Prepare Prompts & Labels
        base_text = self.generate_prompts(row['patient_info'], source_view_name, target_view_name)

        return {
            'fov_img': target_img,
            'cond_img': source_img,
            'base_text': base_text,
            'target_view_name': target_view_name,
            'patient': row['patient_info']
        }
    
if __name__ == "__main__":
    from torchvision import transforms
    import tqdm
    import matplotlib.pyplot as plt

    root_dir =  "/home/DATASETS/OCTA_Dataset/ai-readi/4d80fdbd-065a-4b50-8861-369ec73be154/dataset"

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    train_ds = OCTADataset(csv_file="/home/DATASETS/OCTA_Dataset/ai-readi/4d80fdbd-065a-4b50-8861-369ec73be154/cirrus_multi_view_test_split.csv", 
                           root_dir=root_dir, mode='test')
    print('Test dataset', len(train_ds), '\n')

    # train_dl = torch.utils.data.DataLoader(train_ds, batch_size=3, shuffle=True)
    # for sample in train_dl:
    #     cond = sample['cond_img'][0]
    #     fov = sample['fov_img'][0]
    #     fov_text = sample['base_text'][0]

    #     cond = (cond - cond.min()) / (cond.max() - cond.min())
    #     fov = (fov - fov.min()) / (fov.max() - fov.min())

    #     plt.figure(figsize=(9,4))
    #     plt.subplot(1,2,1);plt.imshow(np.transpose(cond.numpy(), (1, 2, 0)), cmap='gray');plt.title('Conditioned Image');plt.axis('off')
    #     plt.subplot(1,2,2);plt.imshow(np.transpose(fov.numpy(), (1, 2, 0)), cmap='gray');plt.title('FOV Image');plt.axis('off')
    #     plt.suptitle(fov_text)
    #     plt.tight_layout()
    #     plt.show()
    #     plt.savefig('sample.png')
    #     break