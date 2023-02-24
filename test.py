import time
from options.test_options import TestOptions
from data.custom_dataset_data_loader import CreateDataLoader
from model.custom_model import WaterDrop
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
vid_test_loader_syn, vid_test_loader_real = CreateDataLoader(opt)

model = WaterDrop()
model.initialize(opt)

if opt.data_type == 'synthetic':
    model.vid_test_syn(vid_test_loader_syn, int(opt.which_iter))
elif opt.data_type == 'real':
    model.vid_test_real(vid_test_loader_real, int(opt.which_iter))
