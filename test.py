import time
from options.test_options import TestOptions
from data.custom_dataset_data_loader import CreateDataLoader
from model.custom_model import RainDrop
from model.eval_tools import eval_PSNR, eval_SSIM
from util import util
import os
import torch
import numpy as np
import cv2

######## for controllable results ########
SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

opt = TestOptions().parse()
vid_test_loader_syn, vid_test_loader_youtube = CreateDataLoader(opt)

model = RainDrop()
model.initialize(opt)

# model.vid_test_syn(vid_test_loader_syn, int(opt.which_iter))
model.vid_test_youtube(vid_test_loader_youtube, int(opt.which_iter))
