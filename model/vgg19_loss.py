import torch
import torch.nn as nn
import torchvision
from torch.nn.parallel import DataParallel

# class VGG19(torch.nn.Module):
#     def __init__(self, device):
#         super(VGG19, self).__init__()
#         blocks = []
#         blocks.append(torchvision.models.vgg19(pretrained=True).features[:4].eval())
#         blocks.append(torchvision.models.vgg19(pretrained=True).features[4:9].eval())
#         blocks.append(torchvision.models.vgg19(pretrained=True).features[9:16].eval())
#         blocks.append(torchvision.models.vgg19(pretrained=True).features[16:23].eval())
#         blocks.append(torchvision.models.vgg19(pretrained=True).features[23:32].eval())
#         for bl in blocks:
#             for p in bl:
#                 p.requires_grad = False
#         self.blocks = torch.nn.ModuleList(blocks)
#         self.transform = torch.nn.functional.interpolate
#         self.vgg_mean = torch.nn.parameter.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1))
#         self.vgg_std = torch.nn.parameter.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1))
#         self.my_mean = torch.nn.parameter.Parameter(torch.tensor([0.5, 0.5, 0.5]).view(-1, 1, 1))
#         self.my_std = torch.nn.parameter.Parameter(torch.tensor([0.5, 0.5, 0.5]).view(-1, 1, 1))

#     def forward(self, input, target=None, feature_layers=[0, 1, 2, 3, 4]):
#         if target is not None:
#             return self.perceptual_loss(input, target, feature_layers)
#         else:
#             input = input * self.my_std + self.my_mean

#             input = (input-self.vgg_mean) / self.vgg_std

#             x = input
#             for i, block in enumerate(self.blocks):
#                 x = block(x)
#                 if i ==2:
#                     target_feature = x

#             return target_feature
    
#     def perceptual_loss(self, input, target, feature_layers):
#         input = input * self.my_std + self.my_mean
#         target = target * self.my_std + self.my_mean
        
#         input = (input-self.vgg_mean) / self.vgg_std
#         target = (target-self.vgg_mean) / self.vgg_std

#         losses = []
#         x = input
#         y = target
#         for i, block in enumerate(self.blocks):
#             x = block(x)
#             y = block(y)
#             if i in feature_layers:
#                 losses.append(torch.nn.L1Loss()(x, y))
        
#         if len(losses) == 5:
#             loss = losses[0] / 2.6 + losses[1] / 4.8 + losses[2] / 3.7 + losses[3] / 5.6 + losses[4] * 10 / 1.5
#         else:
#             loss = losses[0] / 5.6 + losses[1] * 10 / 1.5
#         return loss

class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)        
        h_relu3 = self.slice3(h_relu2)        
        h_relu4 = self.slice4(h_relu3)        
        h_relu5 = self.slice5(h_relu4)                
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()        
        self.vgg = DataParallel(Vgg19().cuda())
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.downsample = nn.AvgPool2d(2, stride=2, count_include_pad=False)

    def forward(self, x, y):
        while x.size()[3] > 1024:
            x, y = self.downsample(x), self.downsample(y)
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())        
        return loss
