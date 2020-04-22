import torch
import torch.nn as nn
import torch.nn.functional as F


class MarginLoss(nn.Module):
    def __init__(self, args):
        super(MarginLoss, self).__init__()
        self.args = args

    def forward(self, v_c, target):
        labels = F.one_hot(target, self.args.num_classes)
        present_error = F.relu(self.args.m_plus - v_c, inplace=True) ** 2  # max(0, m_plus-||v_c||)^2
        absent_error = F.relu(v_c - self.args.m_minus, inplace=True) ** 2  # max(0, ||v_c||-m_minus)^2

        l_c = labels.float() * present_error + self.args.lambda_val * (1. - labels.float()) * absent_error
        loss = l_c.sum(dim=1).mean()

        return loss


class SpreadLoss(nn.Module):
    def __init__(self, args):
        super(SpreadLoss, self).__init__()
        self.num_class = args.num_classes
        self.margin = args.m_min

    def forward(self, ai, target):
        b, E = ai.shape
        assert E == self.num_class

        at = ai[range(b), target]
        at = at.view(b, 1).repeat(1, E)

        zeros = ai.new_zeros(ai.shape)
        loss = torch.max(self.margin - (at - ai), zeros)
        loss = loss ** 2
        loss = loss.sum() / b - self.margin ** 2

        return loss


class CosineLoss(nn.Module):
    def __init__(self, args):
        super(CosineLoss, self).__init__()
        self.args = args
        self.cossim = nn.CosineSimilarity(dim=1, eps=1e-08)

    def forward(self, f, t):
        batch_size, num_head, dim = f.shape
        loss = torch.mean(1 - self.cossim(f.view(-1, dim), t.view(-1, dim)))
        return loss


class ReconstructionLoss(nn.Module):
    def __init__(self):
        super(ReconstructionLoss, self).__init__()

    def forward(self, x, x_recont):
        assert torch.numel(x) == torch.numel(x_recont)
        x = x.view(x_recont.size()[0], -1)
        reconst_loss = torch.mean((x_recont - x) ** 2)

        return reconst_loss