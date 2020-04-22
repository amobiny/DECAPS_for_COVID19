import torch
import numpy as np
import torch.nn.functional as F


def squash(input_tensor, dim=-1, epsilon=1e-7):
    squared_norm = (input_tensor ** 2).sum(dim=dim, keepdim=True)
    safe_norm = torch.sqrt(squared_norm + epsilon)
    scale = squared_norm / (1 + squared_norm)
    unit_vector = input_tensor / safe_norm
    return scale * unit_vector


def coordinate_addition(v, b, h, w, A, B, psize):
    """
        Shape:
            Input:     (b, H*W*A, B, P*P)
            Output:    (b, H*W*A, B, P*P)
    """
    assert h == w
    v = v.view(b, h, w, A, B, psize)
    coor = torch.arange(h, dtype=torch.float32) / h
    coor_h = torch.cuda.FloatTensor(1, h, 1, 1, 1, psize).fill_(0.)
    coor_w = torch.cuda.FloatTensor(1, 1, w, 1, 1, psize).fill_(0.)
    coor_h[0, :, 0, 0, 0, 0] = coor
    coor_w[0, 0, :, 0, 0, 1] = coor
    v = v + coor_h + coor_w
    v = v.view(b, h * w * A, B, psize)
    return v


def dynamic_routing(v, num_iters):
    """
    applies dynamic routing on the input poses x
    :param v: input votes of shape [batch_size, num_in_caps, num_out_caps, P*P]
    :param num_iters: number of routing iterations
    :return:
    """
    batch_size, num_in_caps, num_out_caps, psize = v.size()

    r_pre = torch.zeros(batch_size, num_in_caps, num_out_caps, 1).cuda()
    for i in range(num_iters):
        r = F.softmax(r_pre, dim=2)  # original is dim=2
        s = (r * v).sum(dim=1)
        p_out = squash(s, dim=-1)
        if i != num_iters - 1:
            v_produce_pout = torch.matmul(v.transpose(1, 2), p_out.unsqueeze(-1))
            r_pre = r_pre + v_produce_pout.transpose(1, 2)
    return p_out


def add_patches(x, A, k, psize, stride):
    """
        Shape:
            Input:     (b, H, W, A*P*P)
            Output:    (b, H', W', K, K, A*P*P)
    """
    b, h, w, c = x.shape
    assert h == w
    assert c == A * psize
    oh = ow = int(((h - k) / stride) + 1)
    idxs = [[(h_idx + k_idx)
             for k_idx in range(0, k)]
            for h_idx in range(0, h - k + 1, stride)]
    x = x[:, idxs, :, :]
    x = x[:, :, :, idxs, :]
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x, oh, ow


def transform_view(x, w):
    """
        Input:     (b*H*W, K*K*A, P*P)
        Output:    (b*H*W, K*K*A, B, P*P)
    """
    b, A, psize = x.shape
    _, Aw, B, P, _ = w.shape
    assert psize == P * P

    x = x.view(b, A, 1, P, P)
    w = w.repeat(b, 1, 1, 1, 1)
    x = x.repeat(1, 1, B, 1, 1)
    v = torch.matmul(x, w)
    v = v.view(b, A, B, P * P)
    return v


def get_vector_length(x, dim=-1):
    x_length = (x ** 2).sum(dim=dim) ** 0.5
    return x_length


def inverted_dynamic_routing(v, num_maps, iters):
    if len(v.size()) == 4:
        batch_size, num_maps_hw, num_out_caps, out_cap_dim = v.size()
        num_in_caps = int(num_maps_hw / num_maps)
        v = v.view(batch_size, num_maps, num_in_caps, num_out_caps, out_cap_dim, 1)
    else:
        batch_size, num_maps, num_in_caps, num_out_caps, out_cap_dim, _ = v.shape
    b = torch.zeros(batch_size, num_maps, num_in_caps, num_out_caps, 1, 1).cuda()
    for i in range(iters):

        # c = F.softmax(b, dim=2)

        c = F.softmax(b.view(batch_size, num_maps, num_in_caps * num_out_caps, 1, 1), dim=2)
        c = c.view(batch_size, num_maps, num_in_caps, num_out_caps, 1, 1)

        s = (c * v).sum(dim=1, keepdim=True).sum(dim=2, keepdim=True)
        outputs = squash(s, dim=-2)

        if i != iters - 1:
            outputs_tiled = outputs.repeat(1, num_maps, num_in_caps, 1, 1, 1)
            u_produce_v = torch.matmul(v.transpose(-1, -2), outputs_tiled)
            b = b + u_produce_v

    map_size = int(np.sqrt(num_in_caps))
    x = (c * v).view(batch_size, num_maps, map_size, map_size, num_out_caps, out_cap_dim)
    hams = get_vector_length(x)

    return outputs.squeeze(2).squeeze(1).squeeze(-1), hams



