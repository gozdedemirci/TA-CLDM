# -*- coding: utf-8 -*-
"""
The implementation is borrowed from: https://github.com/HiLab-git/PyMIC
"""
from __future__ import division, print_function

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.uniform import Uniform
import torch.nn.functional as F

def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model

def sparse_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.sparse_(m.weight, sparsity=0.1)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model
    
class ConvBlock(nn.Module):
    """two convolution layers with batch norm and leaky relu"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(ConvBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv_conv(x)


class DownBlock(nn.Module):
    """Downsampling followed by ConvBlock"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, dropout_p)

        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """Upssampling followed by ConvBlock"""

    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p,
                 bilinear=True):
        super(UpBlock, self).__init__()
        self.bilinear = bilinear
        if bilinear:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels1, in_channels2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2))
        x = torch.cat([x, x2], axis=1)
        # x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        self.dropout = self.params['dropout']
        assert (len(self.ft_chns) == 5)
        self.in_conv = ConvBlock(
            self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1 = DownBlock(
            self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2 = DownBlock(
            self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3 = DownBlock(
            self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.down4 = DownBlock(
            self.ft_chns[3], self.ft_chns[4], self.dropout[4])

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        return [x0, x1, x2, x3, x4]
    
class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,
                                  kernel_size=3, padding=1)
        
    def forward(self, feature):
        ### !! Original !!
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        output = self.out_conv(x)
        # rep = self.representation(x)
        return output


class DecoderAux(nn.Module):
    def __init__(self, params):
        super(DecoderAux, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,
                                  kernel_size=3, padding=1)
        
    def forward(self, feature, feature2):
        # import pdb; pdb.set_trace()
        x0 = feature[0]
        x1 = torch.mean(torch.stack([feature[1], feature2[1]]), dim=0)
        # x1 = (feature[1] + feature2[1]) / 2
        x2 = feature[2]
        x3 = torch.mean(torch.stack([feature[3], feature2[3]]), dim=0)
        # x3 = (feature[3] + feature2[3]) / 2
        x4 = feature[4]

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        output = self.out_conv(x)
        # rep = self.representation(x)
        return output

    # def forward(self, feature):
    #     ### !! Original !!
    #     x0 = feature[0]
    #     x1 = feature[1]
    #     x2 = feature[2]
    #     x3 = feature[3]
    #     x4 = feature[4]

    #     x = self.up1(x4, x3)
    #     x = self.up2(x, x2)
    #     x = self.up3(x, x1)
    #     x = self.up4(x, x0)
    #     output = self.out_conv(x)
    #     # rep = self.representation(x)
    #     return output

def FeatureDropout(x):
    attention = torch.mean(x, dim=1, keepdim=True)
    max_val, _ = torch.max(attention.view(
        x.size(0), -1), dim=1, keepdim=True)
    threshold = max_val * np.random.uniform(0.7, 0.9)
    threshold = threshold.view(x.size(0), 1, 1, 1).expand_as(attention)
    drop_mask = (attention < threshold).float()
    x = x.mul(drop_mask)
    return x

class FeatureNoise(nn.Module):
    def __init__(self, uniform_range=0.3):
        super(FeatureNoise, self).__init__()
        self.uni_dist = Uniform(-uniform_range, uniform_range)

    def feature_based_noise(self, x):
        noise_vector = self.uni_dist.sample(
            x.shape[1:]).to(x.device).unsqueeze(0)
        x_noise = x.mul(noise_vector) + x
        return x_noise

    def forward(self, x):
        x = self.feature_based_noise(x)
        return x

class UNetAux(nn.Module):
    def __init__(self, in_chns, class_num, dropout=False):
        super(UNetAux, self).__init__()

        if dropout:
            params = {'in_chns': in_chns,
                    'feature_chns': [16, 32, 64, 128, 256],
                    'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                    'class_num': class_num,
                    'bilinear': False,
                    'acti_func': 'relu'}
        else:
            params = {'in_chns': in_chns,
                    'feature_chns': [16, 32, 64, 128, 256],
                    'dropout': [0.0, 0.0, 0.0, 0.0, 0.0],
                    'class_num': class_num,
                    'bilinear': False,
                    'acti_func': 'relu'}

        self.encoder = Encoder(params)
        self.decoder = Decoder(params)

        # self.aux_decoder1 = Decoder(params)
        # self.aux_decoder2 = Decoder(params)

        self.aux_decoder1 = DecoderAux(params)
        self.aux_decoder2 = DecoderAux(params)



    def forward(self, x):
        feature = self.encoder(x)
        # output = self.decoder(feature)
        # return output
        main_seg = self.decoder(feature)

        aux1_feature = [FeatureNoise()(i) for i in feature]
        aux2_feature = [FeatureDropout(i) for i in feature]
        aux_seg1 = self.aux_decoder1(aux1_feature, aux2_feature)
        aux_seg2 = self.aux_decoder2(aux2_feature, aux1_feature)

        return main_seg, aux_seg1, aux_seg2
    
class UNetOrg(nn.Module):
    def __init__(self, in_chns, class_num, dropout=False):
        super(UNetOrg, self).__init__()

        if dropout:
            params = {'in_chns': in_chns,
                    'feature_chns': [16, 32, 64, 128, 256],
                    'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                    'class_num': class_num,
                    'bilinear': False,
                    'acti_func': 'relu'}
        else:
            params = {'in_chns': in_chns,
                    'feature_chns': [16, 32, 64, 128, 256],
                    'dropout': [0.0, 0.0, 0.0, 0.0, 0.0],
                    'class_num': class_num,
                    'bilinear': False,
                    'acti_func': 'relu'}

        self.encoder = Encoder(params)
        self.decoder = Decoder(params)

    def forward(self, x):
        feature = self.encoder(x)
        output = self.decoder(feature)
        return output
        

# if __name__ == "__main__":
#     import torch

#     x = torch.randn((2, 3, 400, 400))
#     model = UNetAux(in_chns=3, class_num=2)
#     output, aux1, aux2 = model(x)
#     import pdb; pdb.set_trace()