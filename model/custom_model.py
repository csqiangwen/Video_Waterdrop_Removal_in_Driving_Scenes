from collections import OrderedDict
import torch
import torch.nn as nn
from torch.utils.data.dataset import T
import util.util as image_util
from .base_model import BaseModel
from torch.nn.parallel import DataParallel
from . import networks
import model.util as util
from .vgg19_loss import VGGLoss
from model.eval_tools import eval_PSNR, eval_SSIM
import torch.nn.functional as F
import cv2
from torch.utils.tensorboard import SummaryWriter
import torchvision
import numpy as np
import lpips
from pytorch_msssim import ssim, ms_ssim
import os

# loss_fn_alex = lpips.LPIPS(net='alex').cuda()

# helper function to show an image
# (used in the `plot_classes_preds` function below)
def matplotlib_imshow(img, datatype='image'):
    if datatype == 'image':
        img = img / 2 + 0.5     # unnormalize
        npimg = img.cpu().numpy()
    else:
        img = 1-img
        npimg = img.cpu().numpy()
    
    # return np.transpose(npimg, (1, 2, 0))
    return npimg
        
class RainDrop(BaseModel):
    def name(self):
        return 'RainDrop'

    def initialize(self, opt):
        self.opt = opt
        BaseModel.initialize(self, opt)

        if self.isTrain:
            # For tensorboardX
            self.writer = SummaryWriter()
            # load/define networks
            self.netG = networks.define_G(in_channel=3).to(self.device)
            self.netD = networks.define_D(in_channel=3*self.opt.n_frames).to(self.device)

            # define loss functions
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionL2 = torch.nn.MSELoss(size_average=True)
            self.criterionGAN = networks.GANLoss()
            # self.criterionGAN = networks.AdversarialLoss('hinge').to(self.device)
            self.criterionBCE = torch.nn.BCELoss()
            self.criterionTV = util.TVLoss()
            self.criterionVGG = VGGLoss()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam([{'params': self.netG.single_transformer.parameters(), 'lr':1e-5},
                                                 {'params': self.netG.multi_transformer.parameters(), 'lr':1e-5},
                                                 {'params': self.netG.main_branch.parameters()},
                                                 {'params': self.netG.weight_branch.parameters()},
                                                 {'params': self.netG.res_branch_1.parameters()},
                                                 {'params': self.netG.res_branch_2.parameters()},
                                                 {'params': self.netG.upsample_block.parameters()},
                                                 {'params': self.netG.mask_branch.parameters()},
                                                 {'params': self.netG.side_branch_1.parameters()},
                                                 {'params': self.netG.side_branch_2.parameters()},
                                                 {'params': self.netG.sig_branch.parameters()},
                                                 ],
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=5e-5, betas=(opt.beta1, 0.999))
            self.networks = []
            self.optimizers = []
            self.schedulers = []
            self.networks.append(self.netG)
            self.networks.append(self.netD)
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))
                
            if opt.continue_train:
                which_iter = opt.which_iter
                self.load_states(self.netG, self.optimizer_G, self.schedulers[0], 'G', which_iter)
                self.load_states(self.netD, self.optimizer_D, self.schedulers[1], 'D', which_iter)
                
            if len(self.gpu_ids) > 0:
                self.netG = DataParallel(self.netG)
                self.netD = DataParallel(self.netD)
                
            self.netG.train()
            self.netD.train()
        
        if not self.isTrain:
            # load/define networks
            self.netG = networks.define_G(in_channel=3)

            which_iter = opt.which_iter
            self.load_states_simple(self.netG, 'G', which_iter)
            
            self.netG.eval()
            self.netG = self.netG.cuda()
            
            print('---------- Networks initialized -------------')
            networks.print_network(self.netG)
            print('-----------------------------------------------')


    def img_train_forward(self, data):
        self.img_sample = {}
        self.img_sample['rainy_image'] = data['rainy_image'].to(self.device)
        self.img_sample['clean_image'] = data['clean_image'].to(self.device)
        self.img_sample['rainy_mask'] = data['rainy_mask'].to(self.device)
    
    def vid_train_forward(self, data):
        self.vid_sample = {}
        self.vid_sample['rainy_frames'] = data['rainy_frames'].to(self.device)
        self.vid_sample['rainy_masks'] = data['rainy_masks'].to(self.device)
        self.vid_sample['clean_frames'] = data['clean_frames'].to(self.device)

    
    def D_optim(self, fake, real):
        self.set_requires_grad([self.netD], True)
        self.optimizer_D.zero_grad()
        real_feat = self.netD(real)
        fake_feat = self.netD(fake.detach())
        dis_real_loss = self.criterionGAN(real_feat, True)
        dis_fake_loss = self.criterionGAN(fake_feat, False)
        DLoss = (dis_real_loss + dis_fake_loss) / 2
        DLoss.backward()
        self.optimizer_D.step()
        self.DLoss = DLoss.item()
        return


    def img_optimize_parameters(self, iteration):

        ### input shape [B, T, C, H, W]
        B, T, C, H, W = self.img_sample['clean_image'].shape

        self.optimizer_G.zero_grad()
        
        fake_clean_s_image1, fake_clean_s_image2, fake_clean_image, img_fake_mask = self.netG(self.img_sample['rainy_image'])

        fake_clean_image_final = fake_clean_image * (1-img_fake_mask) + self.img_sample['rainy_image'] * img_fake_mask

        # self.D_optim(fake_clean_image, self.img_sample['clean_image'])
        # self.set_requires_grad([self.netD], False)
        # Loss_GAN = self.criterionGAN(self.netD(fake_clean_image), True) * 5

        Loss_VGG = self.criterionVGG(fake_clean_image.view(B*T, C, H, W), self.img_sample['clean_image'].view(B*T, C, H, W)).mean() * 10

        Loss_VGG_final = self.criterionVGG(fake_clean_image_final.view(B*T, C, H, W), self.img_sample['clean_image'].view(B*T, C, H, W)).mean() * 10
        
        Loss_small = 0
        real_clean_s_image1 = F.interpolate(self.img_sample['clean_image'].view(B*T, C, H, W), scale_factor=0.25)
        real_clean_s_image2 = F.interpolate(self.img_sample['clean_image'].view(B*T, C, H, W), scale_factor=0.5)
        Loss_small += self.criterionL2(fake_clean_s_image1, real_clean_s_image1) * 50
        Loss_small += self.criterionL2(fake_clean_s_image2, real_clean_s_image2) * 25
        
        Loss_mask = self.criterionBCE(img_fake_mask, self.img_sample['rainy_mask']) * 5

        whole_loss = Loss_mask + Loss_small + Loss_VGG + Loss_VGG_final

        # self.writer.add_scalar("Image/Loss_GAN", Loss_GAN, iteration)
        self.writer.add_scalar("Image/Loss_VGG", Loss_VGG, iteration)
        self.writer.add_scalar("Image/Loss_VGG_final", Loss_VGG_final, iteration)
        self.writer.add_scalar("Image/Loss_mask", Loss_mask, iteration)
        self.writer.add_scalar("Image/Loss_small", Loss_small, iteration)
        # self.writer.add_scalar("Image/DLoss", self.DLoss, iteration)

        whole_loss.backward()
        self.optimizer_G.step()

        self.whole_loss = whole_loss.item()
        # self.Loss_GAN = Loss_GAN.item()
        self.Loss_VGG = Loss_VGG.item()
        self.Loss_VGG_final = Loss_VGG_final.item()
        self.Loss_mask = Loss_mask.item()
        self.Loss_small = Loss_small.item()

        self.clean_image = self.img_sample['clean_image'][:, 0]
        self.rainy_image = self.img_sample['rainy_image'][:, 0]
        self.fake_clean_image = fake_clean_image[:, 0].detach()
        self.img_real_mask = self.img_sample['rainy_mask'][:, 0]
        self.img_fake_mask = img_fake_mask[:, 0].detach()

        self.img_psnr = eval_PSNR(image_util.tensor2im(self.fake_clean_image),
                                  image_util.tensor2im(self.clean_image))

        self.writer.add_scalar("Image/PSNR", self.img_psnr, iteration)


    def vid_optimize_parameters(self, iteration):

        ### input shape [B, T, C, H, W]
        B, T, C, H, W = self.vid_sample['clean_frames'].shape

        self.optimizer_G.zero_grad()
        
        fake_clean_s_frames1, fake_clean_s_frames2, fake_clean_frames, vid_fake_masks = self.netG(self.vid_sample['rainy_frames'])
        
        self.D_optim(fake_clean_frames, self.vid_sample['clean_frames'])
        self.set_requires_grad([self.netD], False)
        Loss_GAN = self.criterionGAN(self.netD(fake_clean_frames), True)
        
        Loss_L2andVGG = self.criterionL2(fake_clean_frames, self.vid_sample['clean_frames']) * 25 + \
                        self.criterionVGG(fake_clean_frames.view(B*T, C, H, W), self.vid_sample['clean_frames'].view(B*T, C, H, W)).mean() * 10 +\
                        self.criterionL2(fake_clean_frames*(1-self.vid_sample['rainy_masks']), self.vid_sample['clean_frames']*(1-self.vid_sample['rainy_masks'])) * 25
        
        Loss_small = 0
        real_clean_s_frames1 = F.interpolate(self.vid_sample['clean_frames'].view(B*T, C, H, W), scale_factor=0.25)
        real_clean_s_frames2 = F.interpolate(self.vid_sample['clean_frames'].view(B*T, C, H, W), scale_factor=0.5)
        Loss_small += self.criterionL2(fake_clean_s_frames1, real_clean_s_frames1) * 50
        Loss_small += self.criterionL2(fake_clean_s_frames2, real_clean_s_frames2) * 25

        Loss_mask = self.criterionBCE(vid_fake_masks, self.vid_sample['rainy_masks'].float()) * 10
        
        whole_loss = Loss_GAN + Loss_L2andVGG + Loss_mask + Loss_small

        self.writer.add_scalar("Video/Loss_GAN", Loss_GAN, iteration)
        self.writer.add_scalar("Video/Loss_L2andVGG", Loss_L2andVGG, iteration)
        self.writer.add_scalar("Video/Loss_mask", Loss_mask, iteration)
        self.writer.add_scalar("Video/Loss_small", Loss_small, iteration)
        self.writer.add_scalar("Video/DLoss", self.DLoss, iteration)

        whole_loss.backward()
        self.optimizer_G.step()

        self.whole_loss = whole_loss.item()
        self.Loss_GAN = Loss_GAN.item()
        self.Loss_L2andVGG = Loss_L2andVGG.item()
        self.Loss_mask = Loss_mask.item()
        self.Loss_small = Loss_small.item()

        self.clean_frame = self.vid_sample['clean_frames'][:, 0]
        self.rainy_frame = self.vid_sample['rainy_frames'][:, 0]
        self.vid_real_mask = self.vid_sample['rainy_masks'][:, 0]
        self.fake_clean_frame = fake_clean_frames[:, 0].detach()
        self.vid_fake_mask = vid_fake_masks[:, 0].detach()

        self.vid_psnr = eval_PSNR(image_util.tensor2im(self.fake_clean_frame),
                                  image_util.tensor2im(self.clean_frame))

        self.writer.add_scalar("Video/PSNR", self.vid_psnr, iteration)
        

    def get_current_errors(self, iteration):
        if iteration % 1000 >=0 and iteration % 1000 <100:
            ret_errors = OrderedDict([('whole_loss', self.whole_loss),
                                    #   ('Loss_GAN', self.Loss_GAN),
                                      ('Loss_VGG', self.Loss_VGG),
                                      ('Loss_VGG_final', self.Loss_VGG_final),
                                      ('Loss_mask', self.Loss_mask),
                                      ('Loss_small', self.Loss_small),
                                    #   ('DLoss', self.DLoss),
                                      ('PSNR', self.img_psnr)])
        else:
            ret_errors = OrderedDict([('whole_loss', self.whole_loss),
                                    ('Loss_GAN', self.Loss_GAN),
                                    ('Loss_L2andVGG', self.Loss_L2andVGG),
                                    ('Loss_mask', self.Loss_mask),
                                    ('Loss_small', self.Loss_small),
                                    ('DLoss', self.DLoss),
                                    ('PSNR', self.vid_psnr)])
            
        return ret_errors

    
    def get_current_visuals_train(self, iteration):
        
        if iteration % 1000 >=0 and iteration % 1000 <100:

            img_grid_clean = torchvision.utils.make_grid([self.clean_image[[0]]])
            img_grid_clean = matplotlib_imshow(img_grid_clean[0])

            img_grid_rainy = torchvision.utils.make_grid([self.rainy_image[[0]]])
            img_grid_rainy = matplotlib_imshow(img_grid_rainy[0])

            img_grid_fake_clean = torchvision.utils.make_grid([self.fake_clean_image[[0]]])
            img_grid_fake_clean = matplotlib_imshow(img_grid_fake_clean[0])

            mask_grid = torchvision.utils.make_grid([self.img_fake_mask[[0]]])
            mask_grid = matplotlib_imshow(mask_grid[0], 'mask')
            
            self.writer.add_image('Image/clean_image', img_grid_clean)
            self.writer.add_image('Image/rainy_image', img_grid_rainy)
            self.writer.add_image('Image/fake_clean_image', img_grid_fake_clean)
            self.writer.add_image('Image/fake_mask', mask_grid)

            clean_image = image_util.tensor2im(self.clean_image)
            rainy_image = image_util.tensor2im(self.rainy_image)
            fake_clean_image = image_util.tensor2im(self.fake_clean_image)
            img_real_mask = image_util.tensor2im(self.img_real_mask, 'mask')
            img_fake_mask = image_util.tensor2im(self.img_fake_mask, 'mask')

            ret_visuals = OrderedDict([('clean_image', clean_image),
                                       ('rainy_image', rainy_image),
                                       ('fake_clean_image', fake_clean_image),
                                       ('img_real_mask', img_real_mask),
                                       ('img_fake_mask', img_fake_mask)])

        else:
            frame_grid_clean = torchvision.utils.make_grid([self.clean_frame[[0]]])
            frame_grid_clean = matplotlib_imshow(frame_grid_clean[0])

            frame_grid_rainy = torchvision.utils.make_grid([self.rainy_frame[[0]]])
            frame_grid_rainy = matplotlib_imshow(frame_grid_rainy[0])

            frame_grid_fake_clean = torchvision.utils.make_grid([self.fake_clean_frame[[0]]])
            frame_grid_fake_clean = matplotlib_imshow(frame_grid_fake_clean[0])

            mask_grid = torchvision.utils.make_grid([self.vid_fake_mask[[0]]])
            mask_grid = matplotlib_imshow(mask_grid[0], 'mask')
            
            self.writer.add_image('Video/clean_image', frame_grid_clean)
            self.writer.add_image('Video/rainy_image', frame_grid_rainy)
            self.writer.add_image('Video/fake_clean_image', frame_grid_fake_clean)
            self.writer.add_image('Video/fake_mask', mask_grid)

            clean_frame = image_util.tensor2im(self.clean_frame)
            rainy_frame = image_util.tensor2im(self.rainy_frame)
            fake_clean_frame = image_util.tensor2im(self.fake_clean_frame)
            vid_real_mask = image_util.tensor2im(self.vid_real_mask, 'mask')
            vid_fake_mask = image_util.tensor2im(self.vid_fake_mask, 'mask')

            ret_visuals = OrderedDict([('clean_frame', clean_frame),
                                       ('rainy_frame', rainy_frame),
                                       ('vid_real_mask', vid_real_mask),
                                       ('fake_clean_frame', fake_clean_frame),
                                       ('vid_fake_mask', vid_fake_mask)])

        return ret_visuals

    def get_neighbour_indecies(self, index, f_num, list_len):
        frames_index = []
        if index <= int(f_num/2):
            for i in range(f_num):
                frames_index.append(i)
        elif (index+int(f_num/2))>=list_len:
            for i in range(list_len-f_num, list_len):
                frames_index.append(i)
        else:
            for i in range(index-int(f_num/2), index+int(f_num/2)):
                frames_index.append(i)
        return frames_index

    def vid_test_syn(self, vid_test_loader_syn, iteration):

        ### This part is only for test data or test data score computation and reserve
        f = open("SynTest.txt","a+")
        f.write('iteration: %d'%iteration)

        ave_psnr = 0
        ave_psnr_final = 0
        ave_ssim = 0
        ave_ssim_final = 0
        ave_lpips = 0
        ave_lpips_final = 0

        frame_num = 0

        self.netG.eval()

        with torch.no_grad():
            for i, data in enumerate(vid_test_loader_syn):
                vid_len = data['rainy_frames'].shape[1]
                for f_ind in range(vid_len):
                    f_ind_list = self.get_neighbour_indecies(f_ind, self.opt.n_frames, vid_len)
                    _, _, fake_clean_frames, vid_fake_masks = self.netG(data['rainy_frames'][:, f_ind_list].cuda())
                    fake_clean_frames_final = fake_clean_frames * (1-vid_fake_masks) + data['rainy_frames'][:, f_ind_list].cuda() * vid_fake_masks

                    ### Before BGR to RGB
                    # lpips = loss_fn_alex(fake_clean_frames[:, f_ind-f_ind_list[0]].cpu()[-1:,:, :], data['clean_frames'][:, f_ind][-1:,:, :]).item()
                    # lpips_final = loss_fn_alex(fake_clean_frames_final[:, f_ind-f_ind_list[0]].cpu()[-1:,:, :], data['clean_frames'][:, f_ind][-1:,:, :]).item()

                    # ### After BGR to RGB
                    lpips = loss_fn_alex(fake_clean_frames[:, f_ind-f_ind_list[0]], data['clean_frames'][:, f_ind].cuda()).item()
                    # lpips_final = loss_fn_alex(fake_clean_frames_final[:, f_ind-f_ind_list[0]], data['clean_frames'][:, f_ind].cuda()).item()

                    ssim = ms_ssim((fake_clean_frames[:, f_ind-f_ind_list[0]]+1) / 2, (data['clean_frames'][:, f_ind].cuda()+1) / 2, data_range=1, size_average=True)
                    # ssim_final = ms_ssim((fake_clean_frames_final[:, f_ind-f_ind_list[0]]+1) / 2, (data['clean_frames'][:, f_ind].cuda()+1) / 2, data_range=1, size_average=True)

                    fake_clean_frame_np = image_util.tensor2im(fake_clean_frames[:, f_ind-f_ind_list[0]])
                    fake_clean_frame_final_np = image_util.tensor2im(fake_clean_frames_final[:, f_ind-f_ind_list[0]])
                    vid_fake_mask_np = image_util.tensor2im(1-vid_fake_masks[:, f_ind-f_ind_list[0]], 'mask')
                    color_map_np = cv2.applyColorMap(vid_fake_mask_np, cv2.COLORMAP_JET)

                    rainy_frame_np = image_util.tensor2im(data['rainy_frames'][:, f_ind])
                    real_clean_frame_np = image_util.tensor2im(data['clean_frames'][:, f_ind])

                    psnr = eval_PSNR(fake_clean_frame_np, real_clean_frame_np)
                    # psnr_final = eval_PSNR(fake_clean_frame_final_np, real_clean_frame_np)
                    # ssim = eval_SSIM(fake_clean_frame_np, real_clean_frame_np)
                    # ssim_final = eval_SSIM(fake_clean_frame_final_np, real_clean_frame_np)
                    
                    ave_psnr += psnr
                    # ave_psnr_final += psnr_final
                    ave_ssim += ssim
                    # ave_ssim_final += ssim_final
                    ave_lpips += lpips
                    # ave_lpips_final += lpips_final

                    # cv2.imwrite('vid_test_syn/%d_fake_clean_frame.png'%frame_num, fake_clean_frame_np[:,:,::-1])
                    # cv2.imwrite('vid_test_syn/%d_fake_clean_frame_final.png'%frame_num, fake_clean_frame_final_np[:,:,::-1])
                    # cv2.imwrite('vid_test_syn/%d_real_clean_frame.png'%frame_num, real_clean_frame_np[:,:,::-1])
                    # cv2.imwrite('vid_test_syn/%d_vid_fake_mask.png'%frame_num, color_map_np)
                    # cv2.imwrite('vid_test_syn/%d_rainy_frame.png'%frame_num, rainy_frame_np[:,:,::-1])

                    cv2.imwrite('vid_test_syn/%d_fake_clean_frame.png'%frame_num, fake_clean_frame_np)
                    cv2.imwrite('vid_test_syn/%d_fake_clean_frame_final.png'%frame_num, fake_clean_frame_final_np)
                    cv2.imwrite('vid_test_syn/%d_real_clean_frame.png'%frame_num, real_clean_frame_np)
                    cv2.imwrite('vid_test_syn/%d_vid_fake_mask.png'%frame_num, color_map_np)
                    cv2.imwrite('vid_test_syn/%d_rainy_frame.png'%frame_num, rainy_frame_np)

                    frame_num += 1
                    print('vid_test_syn/num:%04d frame'%frame_num)

        f.write('\n')
        f.write('Frame_Num: {:d}, Average_psnr: {:4.4f}, Average_psnr_final: {:4.4f}, Average_ssim: {:4.4f}, Average_ssim_final: {:4.4f}, Average_lpips: {:4.4f}, Average_lpips_final: {:4.4f}'.format(
                frame_num,
                ave_psnr/frame_num, ave_psnr_final/frame_num,
                ave_ssim/frame_num, ave_ssim_final/frame_num,
                ave_lpips/frame_num, ave_lpips_final/frame_num))  
        f.write('\n')
        f.write('\n')
        f.close()

        self.netG.train()

        return

    def vid_test_youtube(self, vid_test_loader_youtube, iteration):
        img_num = 0

        self.netG.eval()

        with torch.no_grad():
            for i, data in enumerate(vid_test_loader_youtube):
                vid_len = data['rainy_frames'].shape[1]
                for f_ind in range(vid_len):
                    f_ind_list = self.get_neighbour_indecies(f_ind, self.opt.n_frames, vid_len)
                    _, _, fake_clean_frames, vid_fake_masks = self.netG(data['rainy_frames'][:, f_ind_list].cuda())
                    fake_clean_frames_final = fake_clean_frames * (1-vid_fake_masks) + data['rainy_frames'][:, f_ind_list].cuda() * vid_fake_masks

                    fake_clean_frame_np = image_util.tensor2im(fake_clean_frames[:, f_ind-f_ind_list[0]])
                    fake_clean_frame_final_np = image_util.tensor2im(fake_clean_frames_final[:, f_ind-f_ind_list[0]])
                    vid_fake_mask_np = image_util.tensor2im(vid_fake_masks[:, f_ind-f_ind_list[0]], 'mask')
                    vid_fake_mask_np = np.where(vid_fake_mask_np>127.5, 255, 0)
                    # color_map_np = cv2.applyColorMap(vid_fake_mask_np, cv2.COLORMAP_JET)
                    rainy_frame_np = image_util.tensor2im(data['rainy_frames'][:, f_ind])

                    if os.path.isdir('vid_test_supp/%s'%data['vid_name'][0]):
                        pass
                    else:
                        os.mkdir('vid_test_supp/%s'%data['vid_name'][0])
                    cv2.imwrite('vid_test_supp/%s/%d.png'%(data['vid_name'][0], img_num), fake_clean_frame_np)
                    # cv2.imwrite('vid_test_illu/%d_fake_clean_frame_final.png'%img_num, fake_clean_frame_final_np)
                    # cv2.imwrite('vid_test_illu/%d_vid_fake_mask.png'%img_num, vid_fake_mask_np)
                    # cv2.imwrite('vid_test_illu/%d_rainy_frame.png'%img_num, rainy_frame_np)

                    img_num += 1
                    print('vid_test_supp/num:%04d frame'%img_num)
                    break

        self.save_states_simple(self.netG, 'G', self.gpu_ids, self.device)

        self.netG.train()

        return
        
    def save(self, label):
        self.save_states(self.netG, self.optimizer_G, self.schedulers[0], 'G', label, self.gpu_ids, self.device)
        self.save_states(self.netD, self.optimizer_D, self.schedulers[1], 'D', label, self.gpu_ids, self.device)