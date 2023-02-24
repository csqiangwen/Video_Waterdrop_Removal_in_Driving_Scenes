import os.path
import torch
import torch.utils.data as data
import random
import torchvision.transforms as transforms
import data.util as util
import cv2
from natsort import natsorted
import glob

class WaterDropDataset(data.Dataset):
    def __init__(self, opt):
        super(WaterDropDataset,self).__init__()
        self.opt = opt
        self.rainy_vid_path = os.path.join(self.opt.vid_dataroot, 'github', 'test', 'syn', 'rainy_vid')
        self.rainy_vid_names = natsorted(os.listdir(self.rainy_vid_path))
        self.clean_vid_path = os.path.join(self.opt.vid_dataroot, 'github', 'test', 'syn', 'clean_vid')
        self.clean_vid_names = natsorted(os.listdir(self.clean_vid_path))

        self.transforms = transforms.Compose([transforms.ToTensor(),transforms.Normalize(0.5, 0.5)])

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

    def __getitem__(self, index):
        rainy_frames = []
        clean_frames = []

        vid_name = self.clean_vid_names[index%self.vid_num]
        frame_names = natsorted(os.listdir(os.path.join(self.clean_vid_path, vid_name)))[0:100]
        vid_len = len(frame_names)

        for i in range(vid_len):
            rainy_frame = cv2.imread(os.path.join(self.rainy_vid_path, vid_name, frame_names[i]))
            rainy_frame = cv2.cvtColor(rainy_frame, cv2.COLOR_BGR2RGB)
            rainy_frame = util.modcrop(rainy_frame, 32)

            clean_frame = cv2.imread(os.path.join(self.clean_vid_path, vid_name, frame_names[i]))
            clean_frame = cv2.cvtColor(clean_frame, cv2.COLOR_BGR2RGB)
            clean_frame = util.modcrop(clean_frame, 32)
            
            rainy_frames.append(self.transforms(rainy_frame))
            clean_frames.append(self.transforms(clean_frame))

        rainy_frames = torch.stack(rainy_frames, 0)
        clean_frames = torch.stack(clean_frames, 0)

        return {'rainy_frames': rainy_frames, 'clean_frames': clean_frames}

    def __len__(self):
        return self.vid_num

    def name(self):
        return 'Synthetic Dataset'