import torch
import torch.nn as nn
from caps_layers import PrimCapsLayer, ConvCapsLayer, DenseCapsLayer
from utils.caps_utils import get_vector_length
from models import *
import numpy as np


class Conv2dSame(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, padding_layer=nn.ReflectionPad2d):
        super().__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        self.net = nn.Sequential(
            padding_layer((ka, kb, ka, kb)),
            nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias))

    def forward(self, x):
        return self.net(x)


class DECAPS(nn.Module):
    def __init__(self, args):
        super(DECAPS, self).__init__()
        self.args = args
        # backbone CNN
        if args.cnn_backbone == 'resnet':
            net = resnet50(pretrained=True).get_features()
            self.num_features = 1024
        elif args.cnn_backbone == 'densenet':
            net = densenet121(pretrained=True, drop_rate=0.1).get_features()
            self.num_features = 1024
        elif args.cnn_backbone == 'inception':
            net = inception_v3(pretrained=True).get_features()
            net.aux_logits = False
            self.num_features = 768

        self.features = net
        self.conv1x1 = Conv2dSame(self.num_features, args.A, 1)
        self.bn1 = nn.BatchNorm2d(num_features=args.A, eps=0.001, momentum=0.1, affine=True)
        self.relu1 = nn.ReLU(inplace=False)
        self.primary_caps = PrimCapsLayer(A=args.A, B=args.B, K=1, P=4, stride=1)
        self.conv_caps1 = ConvCapsLayer(A=args.B, B=args.C, K=3, P=4, stride=1, iters=args.num_iterations)
        self.conv_caps2 = ConvCapsLayer(A=args.C, B=args.D, K=3, P=4, stride=1, iters=args.num_iterations)
        self.dense_caps = DenseCapsLayer(A=args.D, B=args.num_classes, P=4, iters=args.num_iterations)

    def forward(self, imgs, y=None):
        x = self.features(imgs)
        x = self.relu1(self.bn1(self.conv1x1(x)))
        x = self.primary_caps(x)
        x = self.conv_caps1(x)
        x = self.conv_caps2(x)
        x, activation_maps = self.dense_caps(x)

        if y is None:   # test time
            v_length = get_vector_length(x)
            _, y_pred = v_length.max(dim=1)
            y = y_pred

        # Generate Head Activation Maps (HAMs)
        batch_size, NUM_MAPS, H, W, _ = activation_maps.size()
        if self.training:
            # Randomly choose one of activation maps
            k_indices = np.random.randint(NUM_MAPS, size=batch_size)
            activation_map = activation_maps[torch.arange(batch_size), k_indices, :, :, y].to(torch.device("cuda"))
            # (B, H, W)
        else:
            activation_maps_cls = activation_maps[torch.arange(batch_size), :, :, :, y].to(torch.device("cuda"))
            # (B, NUM_MAPS, H, W)
            activation_map = torch.mean(activation_maps_cls, dim=1, keepdim=True)  # (B, 1, H, W)

        # Normalize Activation Map
        activation_map = activation_map.view(batch_size, -1)  # (B, H * W)
        activation_map_max, _ = activation_map.max(dim=1, keepdim=True)  # (B, 1)
        activation_map_min, _ = activation_map.min(dim=1, keepdim=True)  # (B, 1)
        activation_map = (activation_map - activation_map_min) / (activation_map_max - activation_map_min)  # (B, H * W)
        activation_map = activation_map.view(batch_size, 1, H, W)  # (B, 1, H, W)

        return x, activation_map
