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
class WaterDropDataset(data.Dataset):
    def __init__(self, opt):
        super(WaterDropDataset,self).__init__()
        self.opt = opt
        self.rainy_vid_path = os.path.join(self.opt.vid_dataroot, 'train_vid', 'rainy_vid')
        self.rainy_vid_names = natsorted(os.listdir(self.rainy_vid_path))
        
        self.rainy_mask_path = os.path.join(self.opt.vid_dataroot, 'train_vid', 'rainy_mask')
        self.rainy_mask_names = natsorted(os.listdir(self.rainy_mask_path))

        self.clean_vid_path = os.path.join(self.opt.vid_dataroot, 'train_vid', 'clean_vid')
        self.clean_vid_names = natsorted(os.listdir(self.clean_vid_path))

        self.transforms = transforms.Compose([transforms.ToTensor(),transforms.Normalize(0.5, 0.5)])

        self.frame_num = 65868

        self.vid_num = len(self.clean_vid_names)

    def get_neighbour_indecies(self, index, f_num, list_len):
        frames_index = []
        if index <= int(f_num/2):
            for i in range(f_num+1):
                frames_index.append(i)
        elif (index+int(f_num/2))>=list_len:
            for i in range(list_len-f_num-1, list_len):
                frames_index.append(i)
        else:
            for i in range(index-int(f_num/2), index+int(f_num/2)+1):
                frames_index.append(i)
        return frames_index

    def get_random_index(self, length, sample_length):
        # if random.uniform(0, 1) > 0.5:
        #     ref_index = random.sample(range(length), sample_length)
        #     ref_index.sort()
        # else:
        pivot = random.randint(0, length-sample_length)
        ref_index = [pivot+i for i in range(sample_length)]
        return ref_index

    def random_crop_flip(self, img_list, opt):
        h,w = img_list[0].shape[0], img_list[0].shape[1]
        # h_ind = random.randint(0, h-self.opt.loadsize-1)
        h_ind = 0
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

    def random_flip(self, img_list, opt):
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
        ### get clean video frames path first
        type_in = random.randint(0, 1)
        rainy_type = ['rainy_vid', 'rainy_vid_blur']
        mask_type = ['rainy_mask', 'rainy_mask_blur']
        
        rainy_frames = []
        rainy_masks = []
        clean_frames = []

        vid_name = self.clean_vid_names[index%self.vid_num]
        frame_names = natsorted(os.listdir(os.path.join(self.clean_vid_path, vid_name)))
        vid_len = len(frame_names)

        indecies = self.get_random_index(vid_len, self.opt.n_frames)

        for i in indecies:
            rainy_frame = cv2.imread(os.path.join(self.rainy_vid_path.replace('rainy_vid', rainy_type[type_in]), vid_name, frame_names[i]))
            rainy_frame = cv2.cvtColor(rainy_frame, cv2.COLOR_BGR2RGB)

            rainy_mask = cv2.imread(os.path.join(self.rainy_mask_path.replace('rainy_mask', mask_type[type_in]), vid_name, frame_names[i]))[:, :, [0]]

            clean_frame = cv2.imread(os.path.join(self.clean_vid_path, vid_name, frame_names[i]))
            clean_frame = cv2.cvtColor(clean_frame, cv2.COLOR_BGR2RGB)

            rainy_frame, rainy_mask, clean_frame = self.random_flip([rainy_frame, rainy_mask, clean_frame], self.opt)

            rainy_frames.append(self.transforms(rainy_frame))
            rainy_masks.append(1.0-transforms.ToTensor()(rainy_mask))
            clean_frames.append(self.transforms(clean_frame))

        rainy_frames = torch.stack(rainy_frames, 0)
        rainy_masks = torch.stack(rainy_masks, 0)
        clean_frames = torch.stack(clean_frames, 0)

        return {'rainy_frames': rainy_frames, 'rainy_masks': rainy_masks, 'clean_frames': clean_frames}

    def __len__(self):
        return self.frame_num

    def name(self):
        return 'Video RainDrop Dataset'
