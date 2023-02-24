import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import math
import torch.nn.functional as F


class ImageGradient(nn.Module):
    def __init__(self):
        super(ImageGradient, self).__init__()

        a = np.array([[[[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                       [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                       [[1, 0, -1], [2, 0, -2], [1, 0, -1]]],
                      [[[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                       [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                       [[1, 0, -1], [2, 0, -2], [1, 0, -1]]],
                      [[[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                       [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                       [[1, 0, -1], [2, 0, -2], [1, 0, -1]]]])
        conv1 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False)
        conv1.weight = nn.Parameter(torch.from_numpy(a).float(), requires_grad=False)

        b = np.array([[[[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                       [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                       [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]],
                      [[[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                       [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                       [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]],
                      [[[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                       [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                       [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]]])
        conv2 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False)
        conv2.weight = nn.Parameter(torch.from_numpy(b).float(), requires_grad=False)

        self.conv1 = conv1
        self.conv2 = conv2

    def forward(self, input):
        G_x = self.conv1(input)
        G_y = self.conv2(input)
        
        return torch.cat([G_x, G_y], dim=1)

    
class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]


# ###################################################
# ################## Transformer ####################
# ###################################################


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value):
        scores = torch.matmul(query, key.transpose(-2, -1)
                              ) / math.sqrt(query.size(-1))
        p_attn = F.softmax(scores, dim=-1)
        p_val = torch.matmul(p_attn, value)
        return p_val, p_attn


class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, patchsize, d_model):
        super().__init__()
        self.patchsize = patchsize
        self.query_embedding = nn.Conv2d(
            d_model, d_model, kernel_size=1, padding=0)
        self.value_embedding = nn.Conv2d(
            d_model, d_model, kernel_size=1, padding=0)
        self.key_embedding = nn.Conv2d(
            d_model, d_model, kernel_size=1, padding=0)
        self.output_linear = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True))

        self.attention = Attention()

    def forward(self, x, b, c):
        bt, _, h, w = x.size()
        t = bt // b
        d_k = c // len(self.patchsize)
        output = []
        _query = self.query_embedding(x)
        _key = self.key_embedding(x)
        _value = self.value_embedding(x)

        for (width, height), query, key, value in zip(self.patchsize,
                                                      torch.chunk(_query, len(self.patchsize), dim=1),
                                                      torch.chunk(_key, len(self.patchsize), dim=1),
                                                      torch.chunk(_value, len(self.patchsize), dim=1)):
            out_w, out_h = w // width, h // height
            # 1) embedding and reshape
            query = query.view(b, t, d_k, out_h, height, out_w, width)
            query = query.permute(0, 1, 3, 5, 2, 4, 6).contiguous().view(
                b,  t*out_h*out_w, d_k*height*width)
            key = key.view(b, t, d_k, out_h, height, out_w, width)
            key = key.permute(0, 1, 3, 5, 2, 4, 6).contiguous().view(
                b,  t*out_h*out_w, d_k*height*width)
            value = value.view(b, t, d_k, out_h, height, out_w, width)
            value = value.permute(0, 1, 3, 5, 2, 4, 6).contiguous().view(
                b,  t*out_h*out_w, d_k*height*width)
            '''
            # 2) Apply attention on all the projected vectors in batch.
            tmp1 = []
            for q,k,v in zip(torch.chunk(query, b, dim=0), torch.chunk(key, b, dim=0), torch.chunk(value, b, dim=0)):
                y, _ = self.attention(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0))
                tmp1.append(y)
            y = torch.cat(tmp1,1)
            '''
            y, _ = self.attention(query, key, value)

            # 3) "Concat" using a view and apply a final linear.
            y = y.view(b, t, out_h, out_w, d_k, height, width)
            y = y.permute(0, 1, 4, 2, 5, 3, 6).contiguous().view(bt, d_k, h, w)
            output.append(y)
        output = torch.cat(output, 1)
        x = self.output_linear(output)
        return x


# Standard 2 layerd FFN of transformer
class FeedForward(nn.Module):
    def __init__(self, d_model):
        super(FeedForward, self).__init__()
        # We set d_ff as a default to 2048
        self.conv = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=2, dilation=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class AttentionBlock(nn.Module):

    def __init__(self, patchsize, hidden=128):
        super().__init__()
        self.attention = MultiHeadedAttention(patchsize, d_model=hidden)
        self.feed_forward = FeedForward(hidden)

    def forward(self, x):
        x, b, c = x['x'], x['b'], x['c']
        val = self.attention(x, b, c)
        x = x + val
        x = x + self.feed_forward(x)
        return {'x': x, 'b': b, 'c': c}

     