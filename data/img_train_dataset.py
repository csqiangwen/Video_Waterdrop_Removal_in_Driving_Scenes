import os.path
import torch
import torch.utils.data as data
import random
import torchvision.transforms as transforms
import data.util as util
import cv2
from natsort import natsorted
import glob

## Dataloader for RainDrop dataset
class RainDropDataset(data.Dataset):
    def __init__(self, opt):
        super(RainDropDataset,self).__init__()
        self.opt = opt
        self.rainy_img_path = os.path.join(self.opt.vid_dataroot, 'train_img', 'train', 'data')
        self.rainy_img_names = natsorted(os.listdir(self.rainy_img_path))

        self.clean_img_path = os.path.join(self.opt.vid_dataroot, 'train_img', 'train', 'gt')
        self.clean_img_names = natsorted(os.listdir(self.clean_img_path))

        self.mask_path = os.path.join(self.opt.vid_dataroot, 'train_img', 'train', 'mask')
        self.mask_names = natsorted(os.listdir(self.mask_path))

        self.transforms = transforms.Compose([transforms.ToTensor(),transforms.Normalize(0.5, 0.5)])

        self.img_num = len(self.rainy_img_names)

    def random_crop_flip(self, img_list, opt):
        h,w = img_list[0].shape[0], img_list[0].shape[1]
        h_ind = random.randint(0, h-self.opt.loadsize-1)
        w_ind = random.randint(0, w-self.opt.loadsize-1)
        
        for i, img in enumerate(img_list):
            img_list[i] = img[h_ind:h_ind+opt.loadsize, w_ind:w_ind+opt.loadsize, :]
        
        if random.random() > 0.5:
            for i, img in enumerate(img_list):
                # horizontal
                img_list[i] = cv2.flip(img, 1)
            
        if random.random() > 0.5:
            for i, img in enumerate(img_list):
                # vertical
                img_list[i] = cv2.flip(img, 0)
        return img_list

    def __getitem__(self, index):

        rainy_image = cv2.imread(os.path.join(self.rainy_img_path, self.rainy_img_names[index]))
        rainy_image = cv2.cvtColor(rainy_image, cv2.COLOR_BGR2RGB)

        clean_image = cv2.imread(os.path.join(self.clean_img_path, self.clean_img_names[index]))
        clean_image = cv2.cvtColor(clean_image, cv2.COLOR_BGR2RGB)

        rainy_mask = cv2.imread(os.path.join(self.mask_path, self.mask_names[index]))[:,:,[0]]

        rainy_image, clean_image, rainy_mask = self.random_crop_flip([rainy_image, clean_image, rainy_mask], self.opt)

        rainy_image = self.transforms(rainy_image).unsqueeze(0)
        clean_image = self.transforms(clean_image).unsqueeze(0)
        rainy_mask = 1 - transforms.ToTensor()(rainy_mask).unsqueeze(0)

        return {'rainy_image': rainy_image, 'clean_image': clean_image, 'rainy_mask': rainy_mask}

    def __len__(self):
        return self.img_num

    def name(self):
        return 'Image RainDrop Dataset'
