import torch
import numpy as np


def squash(input_tensor, dim=-1, epsilon=1e-7):
    squared_norm = (input_tensor ** 2).sum(dim=dim, keepdim=True)
    safe_norm = torch.sqrt(squared_norm + epsilon)
    scale = squared_norm / (1 + squared_norm)
    unit_vector = input_tensor / safe_norm
    return scale * unit_vector

#
# def coord_addition(input_tensor, shuffle=False):
#     """
#     adds the coordinates to the input tensor
#     :param input_tensor: tensor of shape [batch_size, capsule_dim, num_capsule_maps, H, W]
#     :return: tensor of shape [batch_size, capsule_dim+2, num_capsule_maps, H, W]
#     """
#     batch_size, _, num_maps, H, W = input_tensor.size()
#     w_offset_vals = np.tile(np.reshape((np.arange(W) + 0.50) / float(W), (1, -1)), (H, 1))
#     h_offset_vals = np.tile(np.reshape((np.arange(H) + 0.50) / float(H), (-1, 1)), (1, W))
#     coordinates = np.stack([w_offset_vals] + [h_offset_vals], axis=0)   # [2, H, W]
#
#     if not shuffle:
#         coordinates = torch.tensor(coordinates[None, :, None, :, :]).repeat(batch_size, 1, num_maps, 1, 1)
#     else:
#         coords = coordinates[None, :, None, :, :]
#         coords_list = []
#         for i in range(num_maps):
#             np.random.shuffle(np.reshape(coords, -1))
#             coords_list.append(coords)
#         coordinates = torch.tensor(np.repeat(np.concatenate(coords_list, axis=2), batch_size, axis=0))
#
#     # coordinates = torch.tensor(coordinates[None, :, None, :, :]).repeat(batch_size, 1, num_maps, 1, 1)
#     # coordinates = torch.zeros_like(coordinates)
#
#     out_tensor = torch.cat((input_tensor, coordinates.float().cuda()), 1)
#     return out_tensor


def coord_addition(input_tensor, norm_coord=False):
    """
    adds the coordinates to the input tensor
    :param input_tensor: tensor of shape [batch_size, num_maps, num_caps, num_cls, caps_dim, 1]
    :return: tensor of shape [batch_size, capsule_dim+2, num_capsule_maps, H, W]
    """
    batch_size, num_maps, num_caps, num_cls, caps_dim, _ = input_tensor.size()
    H = W = int(np.sqrt(num_caps))
    if norm_coord:
        w_offset_vals = np.tile(np.reshape((np.arange(W) + 0.50) / float(W), (1, -1)), (H, 1))
        h_offset_vals = np.tile(np.reshape((np.arange(H) + 0.50) / float(H), (-1, 1)), (1, W))
    else:
        w_offset_vals = np.tile(np.reshape(np.arange(W), (1, -1)), (H, 1))
        h_offset_vals = np.tile(np.reshape(np.arange(H), (-1, 1)), (1, W))
    coords = np.stack([h_offset_vals] + [w_offset_vals], axis=-1)   # [H, W, 2]
    zeros = np.zeros((H, W, caps_dim-2))
    coords = torch.tensor(np.reshape(np.concatenate((zeros, coords), axis=-1), (H*W, caps_dim)))  # [H*W, caps_dim]
    coords = coords[None, None, :, None, :, None].repeat(batch_size, num_maps, 1, num_cls, 1, 1).float().cuda()

    out_tensor = input_tensor + coords
    return out_tensor
