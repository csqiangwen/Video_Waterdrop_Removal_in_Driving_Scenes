import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from model.util import TransformerBlock
import torch.nn.utils.spectral_norm as _spectral_norm
import numpy as np
###############################################################################
# Functions
###############################################################################

def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.niter_decay, gamma=0.5)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def define_G(in_channel, init_type='normal'):
    net = None

    net = Generator(in_channel=in_channel)
    
    return net

def define_D(in_channel):
    netD = None

    netD = MultiscaleDiscriminator(input_nc=in_channel)

    return netD

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


##############################################################################
# Classes
##############################################################################

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def print_network(self):
        if isinstance(self, list):
            self = self[0]
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print('Network [%s] was created. Total number of parameters: %.1f million. '
              'To see the architecture, do print(network).' % (type(self).__name__, num_params / 1000000))

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('InstanceNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight.data, 1.0)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    nn.init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError(
                        'initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)

class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor        
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None        
        gpu_id = input.get_device()
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).cuda(gpu_id).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).cuda(gpu_id).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)                
                loss += self.loss(pred, target_tensor)
            return loss
        else:            
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)


class ResnetBlock(BaseNetwork):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)
        self.init_weights(init_type='xavier')

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                    #    norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class AdversarialLoss(nn.Module):
    r"""
    Adversarial loss
    https://arxiv.org/abs/1711.10337
    """

    def __init__(self, type='nsgan', target_real_label=1.0, target_fake_label=0.0):
        r"""
        type = nsgan | lsgan | hinge
        """
        super(AdversarialLoss, self).__init__()
        self.type = type
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        if type == 'nsgan':
            self.criterion = nn.BCELoss()
        elif type == 'lsgan':
            self.criterion = nn.MSELoss()
        elif type == 'hinge':
            self.criterion = nn.ReLU()

    def __call__(self, outputs, is_real, is_disc=None):
        if self.type == 'hinge':
            if is_disc:
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()
            else:
                return (-outputs).mean()
        else:
            labels = (self.real_label if is_real else self.fake_label).expand_as(
                outputs)
            loss = self.criterion(outputs, labels)
            return loss


class Weight_block_normal(BaseNetwork):
    def __init__(self, in_channel=3, out_channel=512, need_relu=False):
        super(Weight_block_normal, self).__init__()
        self.need_relu = need_relu
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1,
                              padding=0, bias=False)
        # self.norm = nn.InstanceNorm2d(out_channel)
        if need_relu:
            self.relu = nn.ReLU(True)
        
    def forward(self, x):
        if self.need_relu:
            return self.relu(self.conv(x))
        else:
            return self.conv(x)


class Mask_up_block_normal(BaseNetwork):
    def __init__(self, in_channel=3, out_channel=512):
        super(Mask_up_block_normal, self).__init__()
        self.rescale = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1,
                              padding=1, bias=True)
        self.relu = nn.ReLU(True)
        
    def forward(self, x):
        return self.relu(self.conv(self.rescale(x)))


class Downsample_block_normal(BaseNetwork):
    def __init__(self, in_channel=3, out_channel=512):
        super(Downsample_block_normal, self).__init__()
        model = [nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=2,
                              padding=1, bias=True),
                 nn.ReLU(True)]
                #  nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1,
                #               padding=1, bias=True),
                #  nn.ReLU(True)]

        self.model = nn.Sequential(*model)
        self.init_weights(init_type='xavier')
        
    def forward(self, x):
        return self.model(x)
    
    
class Upsample_block_normal(BaseNetwork):
    def __init__(self, in_channel=3, out_channel=512):
        super(Upsample_block_normal, self).__init__()
        model = [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                 nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1,
                              padding=1, bias=True),
                 nn.ReLU(True)]
                #  nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1,
                #               padding=1, bias=True),
                #  nn.ReLU(True)]

        self.model = nn.Sequential(*model)
        self.init_weights(init_type='xavier')
        
    def forward(self, x):
        return self.model(x)

    
### Transformer EmbedNet
class Generator(BaseNetwork):
    def __init__(self, in_channel=3, out_channel=3, ngf=64, n_downsampling=2, single_block_num=1, multi_block_num=4):
        super(Generator, self).__init__()

        single_patchsize = [(2,2)]
        multi_patchsize = [(8,8), (4,4)]
                
        main_branch = [nn.ReflectionPad2d(3),
                       nn.Conv2d(in_channel, ngf, kernel_size=7, padding=0, bias=True),
                       nn.ReLU(inplace=True)]

        for i in range(n_downsampling):
            mult = 2 ** i
            main_branch.append(Downsample_block_normal(ngf * mult, ngf * mult * 2))

        res_block_1 = []
        for i in range(2):
            res_block_1.append(ResnetBlock(ngf * mult * 2, 'reflect', nn.BatchNorm2d, use_dropout=True))

        res_block_2 = []
        for i in range(2):
            res_block_2.append(ResnetBlock(ngf * mult * 2, 'reflect', nn.BatchNorm2d, use_dropout=True))

        weight_branch = [Weight_block_normal(ngf * mult * 2, ngf * mult * 2, need_relu=True),
                         Weight_block_normal(ngf * mult * 2, ngf * mult * 2, need_relu=False)]

        single_blocks = []
        for _ in range(single_block_num):
            single_blocks.append(TransformerBlock(single_patchsize, hidden=ngf * mult * 2))

        multi_blocks = []
        for _ in range(multi_block_num):
            multi_blocks.append(TransformerBlock(multi_patchsize, hidden=ngf * mult * 2))
        
        upsample_block = []
        mask_branch = [nn.ReLU(inplace=True)]
        for i in range(n_downsampling):  # add Upsampling layers
            mult = 2 ** (n_downsampling-i)
            upsample_block.append(Upsample_block_normal(ngf * mult, int(ngf * mult / 2)))
            mask_branch.append(Mask_up_block_normal(ngf * mult, int(ngf * mult / 2)))

        upsample_block += [nn.ReflectionPad2d(3),
                           nn.Conv2d(int(ngf * mult / 2), out_channel, kernel_size=7, padding=0, bias=True),
                           nn.Tanh()]

        mask_branch += [nn.Conv2d(int(ngf * mult / 2), 1, kernel_size=3, padding=1, bias=True),
                        nn.Sigmoid()]

        side_branch_1 = [nn.Conv2d(256, out_channel, kernel_size=3, padding=1, bias=True),
                         nn.Tanh()]
        side_branch_2 = [nn.Conv2d(128, out_channel, kernel_size=3, padding=1, bias=True),
                         nn.Tanh()]

        sig_branch = [nn.Sigmoid()]

            
        self.main_branch = nn.Sequential(*main_branch)
        self.weight_branch = nn.Sequential(*weight_branch)
        self.res_branch_1 = nn.Sequential(*res_block_1)
        self.res_branch_2 = nn.Sequential(*res_block_2)
        self.single_transformer = nn.Sequential(*single_blocks)
        self.multi_transformer = nn.Sequential(*multi_blocks)
        self.upsample_block = nn.Sequential(*upsample_block)
        self.mask_branch = nn.Sequential(*mask_branch)
        self.side_branch_1 = nn.Sequential(*side_branch_1)
        self.side_branch_2 = nn.Sequential(*side_branch_2)
        self.sig_branch = nn.Sequential(*sig_branch)

        self.init_weights(init_type='xavier')
        
    def forward(self, x):
        # The shape of x should be [B, T, C, H, W]
        b, t, c, h, w = x.shape
        x = x.view(b*t, c, h, w).contiguous()
        rainy_main_feature = self.main_branch(x)
        rainy_main_feature = self.res_branch_1(rainy_main_feature)
        weight_feature = self.weight_branch(rainy_main_feature)
        weight_map = self.sig_branch(weight_feature)
        mask = self.mask_branch(weight_feature).view(b, t, 1, h, w)
        main_feature = rainy_main_feature * weight_map
        _, c, sh, sw = main_feature.shape
        single_main_feature = self.single_transformer({'x':main_feature, 'b':b*t, 'c':c})['x']
        single_main_feature = self.res_branch_2(single_main_feature)
        output_small_1 = self.side_branch_1(single_main_feature)
        merge_main_feature = self.multi_transformer({'x':single_main_feature, 'b':b, 'c':c})['x']
        merge_main_feature = self.upsample_block[0](merge_main_feature)
        output_small_2 = self.side_branch_2(merge_main_feature)
        output = self.upsample_block[1:](merge_main_feature)
        output = output.view(b, t, 3, h, w)

        return output_small_1, output_small_2, output, mask

### from STTN
# class Discriminator(BaseNetwork):
#     def __init__(self, in_channels=3, use_sigmoid=True, use_spectral_norm=True, init_weights=True):
#         super(Discriminator, self).__init__()
#         self.use_sigmoid = use_sigmoid
#         nf = 64

#         self.conv = nn.Sequential(
#             spectral_norm(nn.Conv3d(in_channels=in_channels, out_channels=nf*1, kernel_size=(3, 5, 5), stride=(1, 2, 2),
#                                     padding=1, bias=not use_spectral_norm), use_spectral_norm),
#             # nn.InstanceNorm2d(64, track_running_stats=False),
#             nn.LeakyReLU(0.2, inplace=True),
#             spectral_norm(nn.Conv3d(nf*1, nf*2, kernel_size=(3, 5, 5), stride=(1, 2, 2),
#                                     padding=(1, 2, 2), bias=not use_spectral_norm), use_spectral_norm),
#             # nn.InstanceNorm2d(128, track_running_stats=False),
#             nn.LeakyReLU(0.2, inplace=True),
#             spectral_norm(nn.Conv3d(nf * 2, nf * 4, kernel_size=(3, 5, 5), stride=(1, 2, 2),
#                                     padding=(1, 2, 2), bias=not use_spectral_norm), use_spectral_norm),
#             # nn.InstanceNorm2d(256, track_running_stats=False),
#             nn.LeakyReLU(0.2, inplace=True),
#             spectral_norm(nn.Conv3d(nf * 4, nf * 4, kernel_size=(3, 5, 5), stride=(1, 2, 2),
#                                     padding=(1, 2, 2), bias=not use_spectral_norm), use_spectral_norm),
#             # nn.InstanceNorm2d(256, track_running_stats=False),
#             nn.LeakyReLU(0.2, inplace=True),
#             spectral_norm(nn.Conv3d(nf * 4, nf * 4, kernel_size=(3, 5, 5), stride=(1, 2, 2),
#                                     padding=(1, 2, 2), bias=not use_spectral_norm), use_spectral_norm),
#             # nn.InstanceNorm2d(256, track_running_stats=False),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv3d(nf * 4, nf * 4, kernel_size=(3, 5, 5),
#                       stride=(1, 2, 2), padding=(1, 2, 2))
#         )

#         self.sig = nn.Sequential(nn.Sigmoid())

#         self.init_weights()

#     def forward(self, xs):
#         #B, T, C, H, W = xs.shape
#         xs_t = torch.transpose(xs, 1, 2)
#         feat = self.conv(xs_t)
#         if self.use_sigmoid:
#             feat = self.sig(feat)
#         out = torch.transpose(feat, 1, 2)  # B, T, C, H, W
#         return out


# def spectral_norm(module, mode=True):
#     if mode:
#         return _spectral_norm(module)
#     return module

### from Vid2Vid
class MultiscaleDiscriminator(BaseNetwork):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, 
                 num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat
        ndf_max = 64
     
        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, min(ndf_max, ndf*(2**(num_D-1-i))), n_layers, norm_layer,
                                       getIntermFeat)
            if getIntermFeat:                                
                for j in range(n_layers+2):
                    setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD, 'model'+str(j)))                
            else:
                setattr(self, 'layer'+str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        
        self.init_weights()    

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]            
            for i in range(len(model)):
                result.append(model[i](result[-1]))            
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):        
        num_D = self.num_D
        result = []
        # Tensor shape is [B, T, C, H, W]
        B, T, C, H ,W = input.shape
        input = input.view(B, T*C, H, W)
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.n_layers+2)]
            else:
                model = getattr(self, 'layer'+str(num_D-1-i))                                
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D-1):
                input_downsampled = self.downsample(input_downsampled)                    
        return result


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(BaseNetwork):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

        self.init_weights()            

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input) 
