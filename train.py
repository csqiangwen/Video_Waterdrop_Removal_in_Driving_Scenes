import time
from options.train_options import TrainOptions
from data.custom_dataset_data_loader import CreateDataLoader
from model.custom_model import WaterDrop
from model.eval_tools import eval_PSNR
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

opt = TrainOptions().parse()
vid_train_loader, img_train_loader = CreateDataLoader(opt)

model = WaterDrop()
model.initialize(opt)
total_steps = int(1e10)

def print_current_errors(iter, errors, t, log_name):
    message = '(iters: %d, time: %.3f) ' % (iter, t)
    for k, v in errors.items():
        message += '%s: %.3f ' % (k, v)

    print(message)
    with open(log_name, "a") as log_file:
        log_file.write('%s\n' % message)

start_step = int(opt.which_iter)

img_train_iter = iter(img_train_loader)
vid_train_iter = iter(vid_train_loader)

print_time = 0
for iteration in range(start_step, total_steps):
    iter_start_time = time.time()

    if iteration % 1000 >=0 and iteration % 1000 <100:
        try:
            data = img_train_iter.next()
        except:
            img_train_iter = iter(img_train_loader)
            data = img_train_iter.next()

        model.img_train_forward(data)
        model.img_optimize_parameters(iteration)
    else:
        try:
            data = vid_train_iter.next()
        except:
            vid_train_iter = iter(vid_train_loader)
            data = vid_train_iter.next()
        
        model.vid_train_forward(data)
        model.vid_optimize_parameters(iteration)
    
    print_time += time.time() - iter_start_time

    if iteration % opt.print_freq == 0:
        img_dir = os.path.join(opt.checkpoints_dir, 'images')
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        for label, image_numpy in model.get_current_visuals_train(iteration).items():
            img_path = os.path.join(img_dir, '%s.png' % (label))
            util.save_image(image_numpy, img_path)

        errors = model.get_current_errors(iteration)
        t = print_time #(time.time() - iter_start_time) / opt.batchSize
        log_name = os.path.join(opt.checkpoints_dir, 'loss_log.txt')
        print_current_errors(iteration, errors, t, log_name)

        print('End of interation %d \t Every Print Time Taken: %d sec' %
             (iteration, print_time))
        print_time = 0

    if iteration % opt.save_freq == 0:
            
        print('saving the model at the end of iters %d' %
            (iteration))
        model.save('latest')
        model.save(iteration)
        
        model.update_learning_rate(iteration)
