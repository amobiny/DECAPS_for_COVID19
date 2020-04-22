import torch
import torch.nn as nn
from utils.caps_utils import squash, dynamic_routing, add_patches, transform_view, coordinate_addition, \
    inverted_dynamic_routing


class PrimCapsLayer(nn.Module):
    r"""
    Creates a primary convolutional capsule layer
    that outputs a pose matrix.

    Args:
        A: output of the normal conv layer
        B: number of types of capsules
        K: kernel size of convolution
        P: size of pose matrix is P
        stride: stride of convolution

    Shape:
        input:  (*, A, h, w)
        output: (*, h', w', B*P*P)
        h', w' is computed the same way as convolution layer
        parameter size is: K*K*A*B*P*P
    """
    def __init__(self, A, B, K, P, stride):
        super(PrimCapsLayer, self).__init__()
        self.B = B
        self.psize = P*P
        self.pose = nn.Conv2d(A, B*P*P, K, stride, padding=0)

    def forward(self, x):
        p = self.pose(x)
        batch_size, _, h, w = p.size()
        p = p.permute(0, 2, 3, 1)
        p = p.view(batch_size, h, w, self.B, self.psize)
        p = squash(p)
        p = p.view(batch_size, h, w, self.B * self.psize)
        return p


class ConvCapsLayer(nn.Module):
    r"""Create a convolutional capsule layer
    that transfer capsule layer L to capsule layer L+1
    by Inverted Dynamic routing (IDR).

    Args:
        A: input number of types of capsules
        B: output number on types of capsules
        K: kernel size of convolution
        P: size of pose matrix
        stride: stride of convolution
        iters: number of IDR iterations
        coord_add: use scaled coordinate addition or not
    Shape:
        input:  (*, h,  w, A*(P*P))
        output: (*, h', w', B*(P*P))
        h', w' is computed the same way as convolution layer
        parameter size is: K*K*A*B*P*P
    """
    def __init__(self, A, B, K, P, stride, iters, add_coord=False):
        super(ConvCapsLayer, self).__init__()
        self.A = A
        self.B = B
        self.K = K
        self.P = P
        self.psize = P * P
        self.stride = stride
        self.iters = iters
        self.add_coord = add_coord
        self.weights = nn.Parameter(torch.randn(1, K * K * A, B, P, P))

    def forward(self, x):
        b, h, w, c = x.shape

        # add patches
        p_in, oh, ow = add_patches(x, self.A, self.K, self.psize, self.stride)
        p_in = p_in.view(b * oh * ow, self.K * self.K * self.A, self.psize)

        # transform view
        v = transform_view(p_in, self.weights)

        # coor_add
        if self.add_coord:
            v = coordinate_addition(v, b, oh, ow, self.A, self.B, self.psize)

        p_out = dynamic_routing(v, self.iters)
        p_out = p_out.view(b, oh, ow, self.B * self.psize)
        return p_out


class DenseCapsLayer(nn.Module):
    r"""Create a dense capsule layer
    that transfer capsule layer L to capsule layer L+1
    by Inverted Dynamic routing (IDR).

    Args:
        A: input number of types of capsules
        B: output number on types of capsules
        K: kernel size of convolution
        P: size of pose matrix
        stride: stride of convolution
        iters: number of IDR iterations
        coord_add: use scaled coordinate addition or not
    Shape:
        input:  (*, h,  w, A*(P*P))
        output: (*, h', w', B*(P*P))
        h', w' is computed the same way as convolution layer
        parameter size is: K*K*A*B*P*P
    """
    def __init__(self, A, B, P, iters, add_coord=False):
        super(DenseCapsLayer, self).__init__()
        self.A = A
        self.B = B
        self.P = P
        self.psize = P * P
        self.iters = iters
        self.add_coord = add_coord
        self.weights = nn.Parameter(torch.randn(1, A, 1, B, P, P))

    def forward(self, x):
        b, oh, ow, _ = x.size()     # [b, oh, ow, A*P*P]
        p_in = x.view(b, oh * ow, self.A, self.P, self.P)

        # transform view
        p_in = p_in.transpose(1, 2)
        p_in = p_in[:, :, :, None, :]
        v = torch.matmul(self.weights, p_in)
        v = v.view(b, self.A, oh*ow, self.B, self.psize, 1)

        # Coordinate Addition
        if self.add_coord:
            v = coordinate_addition(v, b, oh, ow, self.A, self.B, self.psize)

        p_out = inverted_dynamic_routing(v, self.A, self.iters)
        return p_out
