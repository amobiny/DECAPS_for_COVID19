import os
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import confusion_matrix
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import Adam
from torch.utils.data import DataLoader

from config import options
from utils.caps_utils import get_vector_length
from decaps import DECAPS
from dataset.covid_ct import COVIDDataSet as data
from utils.eval_utils import compute_metrics, plot_roc

os.environ['CUDA_VISIBLE_DEVICES'] = '3'


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


@torch.no_grad()
def evaluate():
    capsule_net.eval()
    test_loss = np.zeros(3)
    targets, preds_coarse, preds_fine, preds_dist = [], [], [], []

    with torch.no_grad():
        for batch_id, (data, target, _) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda()
            p_coarse, avg_activation_map = capsule_net(data, target)
            p_coarse_length = get_vector_length(p_coarse)
            loss = capsule_loss(p_coarse_length, target)
            targets += [target]
            preds_coarse += [p_coarse_length]
            test_loss[0] += loss

            ##################################
            # Distillation (Localization and Refinement)
            ##################################
            crop_mask = F.upsample_bilinear(avg_activation_map, size=(data.size(2), data.size(3))) > options.theta_c
            crop_images = []
            for batch_index in range(crop_mask.size(0)):
                nonzero_indices = torch.nonzero(crop_mask[batch_index, 0, ...])
                height_min = nonzero_indices[:, 0].min()
                height_max = nonzero_indices[:, 0].max()
                width_min = nonzero_indices[:, 1].min()
                width_max = nonzero_indices[:, 1].max()
                crop_images.append(F.upsample_bilinear(
                    data[batch_index:batch_index + 1, :, height_min:height_max, width_min:width_max],
                    size=options.img_h))
            crop_images = torch.cat(crop_images, dim=0)

            p_fine, _ = capsule_net(crop_images, target)
            p_fine_length = get_vector_length(p_fine)
            loss = capsule_loss(p_fine_length, target)
            preds_fine += [p_fine_length]
            test_loss[1] += loss

            p_dist = (p_coarse + p_fine) / 2
            p_dist_length = get_vector_length(p_dist)
            test_loss[2] += capsule_loss(p_dist_length, target)
            preds_dist += [p_dist_length]

        test_loss /= (batch_id + 1)
        # compute metrics
        metrics_coarse = compute_metrics(torch.cat(preds_coarse).cpu(), torch.cat(targets).cpu())
        metrics_fine = compute_metrics(torch.cat(preds_fine).cpu(), torch.cat(targets).cpu())
        metrics_dist = compute_metrics(torch.cat(preds_dist).cpu(), torch.cat(targets).cpu())

        plot_roc(metrics_coarse['fpr'], metrics_coarse['tpr'], metrics_coarse['auc'], save_dir+'/coarse.svg')
        plot_roc(metrics_fine['fpr'], metrics_fine['tpr'], metrics_fine['auc'], save_dir+'/fine.svg')
        plot_roc(metrics_dist['fpr'], metrics_dist['tpr'], metrics_dist['auc'], save_dir+'/distilled.svg')

        # display
        log_string(" - (Coarse-grained)     loss: {0:.4f}, acc: {1:.02%}, auc: {2:.02%}"
                   .format(test_loss[0], metrics_coarse['acc'], metrics_coarse['auc']))
        log_string(" - (Fine-grained)       loss: {0:.4f}, acc: {1:.02%}, auc: {2:.02%}"
                   .format(test_loss[1], metrics_fine['acc'], metrics_fine['auc']))
        log_string(" - (Distilled)          loss: {0:.4f}, acc: {1:.02%}, auc: {2:.02%}"
                   .format(test_loss[2], metrics_dist['acc'], metrics_dist['auc']))

        # compute the confusion matrix
        y_true = torch.cat(targets).cpu().numpy()
        y_pred = torch.cat(preds_dist).cpu().numpy().argmax(axis=1)
        conf_mat = confusion_matrix(y_true, y_pred)
        print(conf_mat)
        conf_mat = conf_mat.flatten()
        conf_mat_text = ''
        for val in conf_mat:
            conf_mat_text = conf_mat_text + str(val) + ', '
        log_string(conf_mat_text)


if __name__ == '__main__':
    ##################################
    # Initialize saving directory
    ##################################
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    iter_num = options.load_model_path.split('/')[-1].split('.')[0]

    save_dir = os.path.dirname(os.path.dirname(options.load_model_path))
    img_dir = os.path.join(save_dir, 'imgs')
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    viz_dir = os.path.join(img_dir, iter_num+'_{}'.format(options.theta_c))
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)

    LOG_FOUT = open(os.path.join(save_dir, 'log_inference.txt'), 'w')
    LOG_FOUT.write(str(options) + '\n')

    # bkp of inference
    os.system('cp {}/inference_attention_with_crop.py {}'.format(BASE_DIR, save_dir))

    ##################################
    # Create the model
    ##################################
    capsule_net = DECAPS(options)
    log_string('Model Generated.')
    log_string("Number of trainable parameters: {}".format(sum(param.numel() for param in capsule_net.parameters())))

    ##################################
    # Use cuda
    ##################################
    cudnn.benchmark = True
    capsule_net.cuda()
    capsule_net = nn.DataParallel(capsule_net)

    ##################################
    # Load the trained model
    ##################################
    ckpt = options.load_model_path
    checkpoint = torch.load(ckpt)
    state_dict = checkpoint['state_dict']

    # Load weights
    capsule_net.load_state_dict(state_dict)
    log_string('Model successfully loaded from {}'.format(ckpt))
    if 'feature_center' in checkpoint:
        feature_center = checkpoint['feature_center'].to(torch.device("cuda"))
        log_string('feature_center loaded from {}'.format(ckpt))

    ##################################
    # Loss and Optimizer
    ##################################
    if options.loss_type == 'margin':
        from utils.loss_utils import MarginLoss

        capsule_loss = MarginLoss(options)
    elif options.loss_type == 'spread':
        from utils.loss_utils import SpreadLoss

        capsule_loss = SpreadLoss(options)
    elif options.loss_type == 'cross-entropy':
        capsule_loss = nn.CrossEntropyLoss()

    optimizer = Adam(capsule_net.parameters(), lr=options.lr, betas=(options.beta1, 0.999))
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)

    ##################################
    # Load dataset
    ##################################
    test_dataset = data(mode='test', args=options)
    test_loader = DataLoader(test_dataset, batch_size=options.batch_size,
                             shuffle=False, num_workers=options.workers, drop_last=False)
    ##################################
    # TESTING
    ##################################
    log_string('')
    log_string('Start Testing')

    evaluate()
