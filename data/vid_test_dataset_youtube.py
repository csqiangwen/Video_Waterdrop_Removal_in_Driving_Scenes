import os.path
import torch
import torch.utils.data as data
import random
import torchvision.transforms as transforms
import data.util as util
import cv2
from natsort import natsorted
import glob

# Good result 12_small

## Dataloader for RainDrop dataset
class RainDropDataset(data.Dataset):
    def __init__(self, opt):
        super(RainDropDataset,self).__init__()
        self.opt = opt
        self.rainy_vid_path = os.path.join(self.opt.vid_dataroot, 'DREYE_new', 'supp', 'vid_cut')
        # self.rainy_vid_names = natsorted(os.listdir(self.rainy_vid_path))
        self.rainy_vid_names = ['12_1']
        # self.rainy_vid_path = os.path.join('/disk1/wenqiang/Documents/data/RainDrop', 'youtube-video')
        # # self.rainy_vid_names = [str(i)+'_small'for i in range(23)]
        # self.rainy_vid_names = ['0_small']

        self.transforms = transforms.Compose([transforms.ToTensor(),transforms.Normalize(0.5, 0.5)])

        self.vid_num = len(self.rainy_vid_names)

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
        if random.uniform(0, 1) > 0.5:
            ref_index = random.sample(range(length), sample_length)
            ref_index.sort()
        else:
            pivot = random.randint(0, length-sample_length)
            ref_index = [pivot+i for i in range(sample_length)]
        return ref_index

    def __getitem__(self, index):
        ### get clean video frames path first
        rainy_frames = []
        frame_names_record = []

        vid_name = self.rainy_vid_names[index%self.vid_num]
        frame_names = natsorted(os.listdir(os.path.join(self.rainy_vid_path, vid_name)))
        vid_len = len(frame_names)

        for i in range(vid_len):
            rainy_frame = cv2.imread(os.path.join(self.rainy_vid_path, vid_name, frame_names[i]))
            rainy_frame = cv2.cvtColor(rainy_frame, cv2.COLOR_BGR2RGB)
            # rainy_frame = cv2.resize(rainy_frame, (0,0), fx=0.75, fy=0.75, interpolation=cv2.INTER_NEAREST)
            # rainy_frame = util.modcrop(rainy_frame, 32)

            rainy_frames.append(self.transforms(rainy_frame))
            frame_names_record.append(os.path.join(self.rainy_vid_path, vid_name, frame_names[i]))

        rainy_frames = torch.stack(rainy_frames, 0)

        return {'rainy_frames': rainy_frames, 'frame_names': frame_names_record, 'vid_name': vid_name}

    def __len__(self):
        return self.vid_num

    def name(self):
        return 'Youtube Dataset'