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
        self.rainy_img_path = os.path.join(self.opt.vid_dataroot, 'test_a', 'data')
        self.rainy_img_names = natsorted(os.listdir(self.rainy_img_path))

        self.clean_img_path = os.path.join(self.opt.vid_dataroot, 'test_a', 'gt')
        self.clean_img_names = natsorted(os.listdir(self.clean_img_path))

        self.transforms = transforms.Compose([transforms.ToTensor(),transforms.Normalize(0.5, 0.5)])

        self.img_num = len(self.rainy_img_names)

    def __getitem__(self, index):

        rainy_image = cv2.imread(os.path.join(self.rainy_img_path, self.rainy_img_names[index]))
        rainy_image = util.modcrop(rainy_image, 32)
        rainy_image = self.transforms(rainy_image).unsqueeze(0)[:,:,:480,:704]

        clean_image = cv2.imread(os.path.join(self.clean_img_path, self.clean_img_names[index]))
        clean_image = util.modcrop(clean_image, 32)
        clean_image = self.transforms(clean_image).unsqueeze(0)[:,:,:480,:704]

        return {'rainy_image': rainy_image, 'clean_image': clean_image}

    def __len__(self):
        return self.img_num

    def name(self):
        return 'Image RainDrop Dataset'