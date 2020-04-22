import os
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import confusion_matrix
from datetime import datetime
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import Adam
from torch.utils.data import DataLoader

from config import options
from utils.eval_utils import compute_accuracy, compute_metrics
from utils.logger_utils import Logger
from utils.caps_utils import get_vector_length
from decaps import DECAPS
from dataset.covid_ct import COVIDDataSet as data

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


def train():
    global_step = 0
    best_loss = 100
    best_acc = 0
    best_auc = 0

    for epoch in range(options.epochs):
        log_string('**' * 30)
        log_string('Training Epoch %03d, Learning Rate %g' %
                   (epoch + 1, optimizer.param_groups[0]['lr']))
        capsule_net.train()

        total_loss_ = np.zeros(4)
        targets, preds_coarse, preds_fine, preds_drop, preds_dist = [], [], [], [], []

        # increments the margin for spread loss
        if options.loss_type == 'spread' and (epoch + 1) % options.n_eps_for_m == 0 and epoch != 0:
            capsule_loss.margin += options.m_delta
            capsule_loss.margin = min(capsule_loss.margin, options.m_max)
            log_string(' *------- Margin increased to {0:.1f}'.format(capsule_loss.margin))

        for batch_id, (data, target, _) in enumerate(train_loader):
            global_step += 1
            data, target = data.cuda(), target.cuda()
            p_coarse, activation_map = capsule_net(data, target)
            p_coarse_length = get_vector_length(p_coarse)
            loss = capsule_loss(p_coarse_length, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            targets += [target]
            preds_coarse += [p_coarse_length]
            total_loss_[0] += loss.item()

    ##################################
    # PEEKABOO Training
    ##################################

            ##################################
            # Patch Cropping
            ##################################
            with torch.no_grad():
                crop_mask = F.upsample_bilinear(activation_map, size=(data.size(2), data.size(3))) > options.theta_c
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

            crop_images = torch.cat(crop_images, dim=0).cuda()
            p_fine, _ = capsule_net(crop_images, target)
            p_fine_length = get_vector_length(p_fine)
            loss = capsule_loss(p_fine_length, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds_fine += [p_fine_length.float()]
            total_loss_[1] += loss.item()

            ##################################
            # Patch Dropping
            ##################################
            with torch.no_grad():
                drop_mask = F.upsample_bilinear(activation_map, size=(data.size(2), data.size(3))) <= options.theta_d
                drop_images = data * drop_mask.float()

            # drop images forward
            p_drop, _ = capsule_net(drop_images.cuda(), target)
            p_drop_length = get_vector_length(p_drop)
            loss = capsule_loss(p_drop_length, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            preds_drop += [p_drop_length.float()]
            total_loss_[2] += loss.item()

            ##################################
            # Distillation
            ##################################
            p_dist = (p_coarse + p_fine) / 2
            p_dist_length = get_vector_length(p_dist)
            loss = capsule_loss(p_dist_length, target)

            preds_dist += [p_dist_length]
            total_loss_[3] += loss.item()

            ##################################
            # Displaying and Validation
            ##################################

            if (batch_id + 1) % options.disp_freq == 0:
                total_loss_ = total_loss_ / options.disp_freq
                train_acc_coarse = compute_accuracy(torch.cat(preds_coarse), torch.cat(targets))
                train_acc_fine = compute_accuracy(torch.cat(preds_fine), torch.cat(targets))
                train_acc_drop = compute_accuracy(torch.cat(preds_drop), torch.cat(targets))
                train_acc_dist = compute_accuracy(torch.cat(preds_dist), torch.cat(targets))

                log_string("epoch: {0}, step: {1}, (Coarse-grained): loss: {2:.4f} acc: {3:.02%}, "
                           "(Fine-grained): loss: {4:.4f} acc: {5:.02%}, "
                           "(Drop): loss: {6:.4f} acc: {7:.02%}, "
                           "(Distilled): loss: {8:.4f} acc: {9:.02%}"
                           .format(epoch + 1, batch_id + 1,
                                   total_loss_[0], train_acc_coarse,
                                   total_loss_[1], train_acc_fine,
                                   total_loss_[2], train_acc_drop,
                                   total_loss_[3], train_acc_dist))
                info = {'loss/coarse-grained': total_loss_[0],
                        'loss/fine-grained': total_loss_[1],
                        'loss/drop': total_loss_[2],
                        'loss/distilled': total_loss_[3],
                        'accuracy/coarse-grained': train_acc_coarse,
                        'accuracy/fine-grained': train_acc_fine,
                        'accuracy/drop': train_acc_drop,
                        'accuracy/distilled': train_acc_dist}

                for tag, value in info.items():
                    train_logger.scalar_summary(tag, value, global_step)
                margin_loss_, reg_loss_, total_loss_ = 0, 0, np.zeros(4)
                targets, preds_coarse, preds_fine, preds_drop, preds_dist = [], [], [], [], []

            if (batch_id + 1) % options.val_freq == 0:
                log_string('--' * 30)
                log_string('Evaluating at step #{}'.format(global_step))
                best_loss, best_acc, best_auc = evaluate(best_loss=best_loss,
                                                         best_acc=best_acc,
                                                         best_auc=best_auc,
                                                         global_step=global_step)
                capsule_net.train()


@torch.no_grad()
def evaluate(**kwargs):
    best_loss = kwargs['best_loss']
    best_acc = kwargs['best_acc']
    best_auc = kwargs['best_auc']
    global_step = kwargs['global_step']

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

        # check for improvement
        loss_str, acc_str, auc_str = '', '', ''
        if test_loss[2] <= best_loss:
            loss_str, best_loss = '(improved)', test_loss[2]
        if metrics_dist['acc'] >= best_acc:
            acc_str, best_acc = '(improved)', metrics_dist['acc']
        if metrics_dist['auc'] >= best_auc:
            auc_str, best_auc = '(improved)', metrics_dist['auc']

        # display
        log_string(" - (Coarse-grained)     loss: {0:.4f}, acc: {1:.02%}, auc: {2:.02%}"
                   .format(test_loss[0], metrics_coarse['acc'], metrics_coarse['auc']))
        log_string(" - (Fine-grained)       loss: {0:.4f}, acc: {1:.02%}, auc: {2:.02%}"
                   .format(test_loss[1], metrics_fine['acc'], metrics_fine['auc']))
        log_string(" - (Distilled)          loss: {0:.4f} {1}, acc: {2:.02%}{3}, auc: {4:.02%}{5}"
                   .format(test_loss[2], loss_str, metrics_dist['acc'], acc_str, metrics_dist['auc'], auc_str))
        # write to TensorBoard
        info = {'loss/coarse-grained': test_loss[0],
                'loss/fine-grained': test_loss[1],
                'loss/distilled': test_loss[2],
                'accuracy/coarse-grained': metrics_coarse['acc'],
                'accuracy/fine-grained': metrics_fine['acc'],
                'accuracy/distilled': metrics_dist['acc'],
                'AUC/coarse-grained': metrics_coarse['auc'],
                'AUC/fine-grained': metrics_fine['auc'],
                'AUC/distilled': metrics_dist['auc']}
        for tag, value in info.items():
            test_logger.scalar_summary(tag, value, global_step)

        # save checkpoint model
        state_dict = capsule_net.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()
        save_path = os.path.join(model_dir, '{}.ckpt'.format(global_step))
        torch.save({
            'global_step': global_step,
            'acc': metrics_dist['acc'],
            'auc': metrics_dist['auc'],
            'save_dir': model_dir,
            'state_dict': state_dict},
            save_path)
        log_string('Model saved at: {}'.format(save_path))
        log_string('--' * 30)
        return best_loss, best_acc, best_auc


if __name__ == '__main__':
    ##################################
    # Initialize saving directory
    ##################################
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    save_dir = options.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_dir = os.path.join(save_dir, datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(save_dir)

    LOG_FOUT = open(os.path.join(save_dir, 'log_train.txt'), 'w')
    LOG_FOUT.write(str(options) + '\n')

    model_dir = os.path.join(save_dir, 'models')
    logs_dir = os.path.join(save_dir, 'tf_logs')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # bkp of model def
    os.system('cp {}/decaps.py {}'.format(BASE_DIR, save_dir))
    # bkp of train procedure
    os.system('cp {}/train.py {}'.format(BASE_DIR, save_dir))
    os.system('cp {}/config.py {}'.format(BASE_DIR, save_dir))

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
    # Load the pre-trained model (uncomment to continue training)
    ##################################
    # ckpt = options.load_model_path
    # checkpoint = torch.load(ckpt)
    # state_dict = checkpoint['state_dict']
    #
    # # Load weights
    # capsule_net.load_state_dict(state_dict)
    # log_string('Model successfully loaded from {}'.format(ckpt))
    # if 'feature_center' in checkpoint:
    #     feature_center = checkpoint['feature_center'].to(torch.device("cuda"))
    #     log_string('feature_center loaded from {}'.format(ckpt))
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
    os.system('cp {}/dataset/covid_ct.py {}'.format(BASE_DIR, save_dir))

    train_dataset = data(mode='train', args=options)
    train_loader = DataLoader(train_dataset, batch_size=options.batch_size,
                              shuffle=True, num_workers=options.workers, drop_last=False)
    test_dataset = data(mode='test', args=options)
    test_loader = DataLoader(test_dataset, batch_size=options.batch_size,
                             shuffle=False, num_workers=options.workers, drop_last=False)

    ##################################
    # TRAINING
    ##################################
    log_string('Loss: {}, ADD_GAN: {}, #CLASSES: {}'.format(options.loss_type, options.add_gan, options.num_classes))
    log_string('Start training {}: Total epochs: {}, Batch size: {}, Training size: {}, Validation size: {}'.
               format(save_dir, options.epochs, options.batch_size, len(train_dataset), len(test_dataset)))
    train_logger = Logger(os.path.join(logs_dir, 'train'))
    test_logger = Logger(os.path.join(logs_dir, 'test'))

    train()
