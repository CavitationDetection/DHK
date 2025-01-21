import pandas as pd
import numpy as np
import torch
import os
from PIL import Image
import torchvision.transforms as transforms
import pickle
from torch.utils.data import DataLoader



transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
])

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, opt, transform = None):
        """
        Args:
            opt: An options object containing configuration.
            transform: A list of image transformations.
        """
        self.opt = opt
        self.root = opt.test_root
        self.label = opt.test_label_path
        self.transform = transform
        self.img_files = os.listdir(self.root)
        self.img_id_list = np.array([int(file.split('.')[0]) for file in self.img_files])
        self.labels = np.array(pd.read_csv(self.label, header=0)['label'])[self.img_id_list]

    def load_and_preprocess_image(self, item):
        img = Image.open(os.path.join(self.root, self.img_files[item])).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img
    
    def __getitem__(self, item):
        img = self.load_and_preprocess_image(item).type(torch.float32)
        return img, self.labels[item]
        
    def __len__(self):
        return len(self.img_files)


if __name__ == "__main__":
    class Options:
        def __init__(self):
            self.test_root = './TestData'  
            self.test_label_path = './Label/test_label.csv'  
    opt = Options()

    dataset = TestDataset(opt, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Iterate over a few batches and print the results
    for i, (images, labels) in enumerate(dataloader):
        print(f"Batch {i+1}")
        print(f"Images shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Labels: {labels}")
        
        if i == 2:  # Only display the first 3 batches for brevity
            break