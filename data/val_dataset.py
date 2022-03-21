import os.path
import torch.utils.data as data
import random
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np
import torch
import data.util as util
import cv2

## Dataloader for RainDrop dataset
class RainDropDataset(data.Dataset):
    def __init__(self, opt):
        super(RainDropDataset,self).__init__()
        self.opt = opt
        self.rainy_imgs_path = os.path.join(self.opt.dataroot, 'data')
        self.rainy_imgs_names = sorted(os.listdir(self.rainy_imgs_path))
        self.gts_path = os.path.join(self.opt.dataroot, 'gt')
        self.gts_names = sorted(os.listdir(self.gts_path))

        self.transforms = transforms.Compose([transforms.ToTensor(),transforms.Normalize(0.5, 0.5)])

        self.test_imgs_names = []
        self.test_gts_names = []
        
        for i in [5, 305, 605, 805]:
            self.test_imgs_names.append(self.rainy_imgs_names[i])
            self.test_gts_names.append(self.gts_names[i])

        self.imgs_n = len(self.test_imgs_names)

    def crop(self, np_img, cropsize=256):
        h, w = np_img.shape[0], np_img.shape[1]
        h, w = random.randint(0, h-cropsize), random.randint(0, w-cropsize)
        crop_img = np_img[h:h+cropsize, w:w+cropsize]
        # crop_img = cv2.resize(crop_img, (size,size), cv2.INTER_NEAREST)
        return crop_img


    def __getitem__(self, index):
        rainy_img = cv2.imread(os.path.join(self.rainy_imgs_path, self.test_imgs_names[index%self.imgs_n]))
        gt_img = cv2.imread(os.path.join(self.gts_path, self.test_gts_names[index%self.imgs_n]))

        rainy_img = util.modcrop(rainy_img, 32)
        gt_img = util.modcrop(gt_img, 32)
        
        self.rainy_img = self.transforms(rainy_img)
        self.gt_img = self.transforms(gt_img)

        return {'rainy_img': self.rainy_img, 'gt_img':self.gt_img}

    def __len__(self):
        return self.imgs_n

    def name(self):
        return 'RainDrop Dataset'