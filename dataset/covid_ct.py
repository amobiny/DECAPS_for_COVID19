from torch.utils.data import Dataset
import os
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import glob


def oversample(imgs, labels, rep=2):
    covid_idxs = np.where(labels == 0)[0]
    covid_imgs = imgs[covid_idxs]
    covid_labels = labels[covid_idxs]
    covid_imgs_rep = np.repeat(covid_imgs, rep)
    covid_labels_rep = np.repeat(covid_labels, rep)
    imgs_aug = np.append(imgs, covid_imgs_rep)
    labels_aug = np.append(labels, covid_labels_rep)
    return imgs_aug, labels_aug


def gan_aug(image_paths, labels, gan_dir, fake_reps=1):
    fake_img_paths = np.array(glob.glob(gan_dir + '/*.png'))
    fake_labels = np.ones(len(fake_img_paths)).astype(int)

    fake_img_paths = np.repeat(fake_img_paths, fake_reps)
    fake_labels = np.repeat(fake_labels, fake_reps)

    image_paths = np.append(image_paths, fake_img_paths)
    labels = np.append(labels, fake_labels)
    return image_paths, labels


class COVIDDataSet(Dataset):
    def __init__(self, mode, args):
        """
        image_list_file: path to the file containing images with corresponding labels.
        transform: optional transform to be applied on a sample.
        Upolicy: name the policy with regard to the uncertain labels
        """
        self.input_size = (args.img_h, args.img_w)
        root = args.data_root
        if mode == 'train':
            self.is_train = True
            self.data_dir = os.path.join(root, 'Train')
        else:
            self.is_train = False
            self.data_dir = os.path.join(root, 'Test')

        image_paths = glob.glob(os.path.join(self.data_dir, 'NonCOVID', '*'))
        labels = np.zeros(len(image_paths))
        image_paths += glob.glob(os.path.join(self.data_dir, 'COVID', '*'))
        image_paths = np.array(image_paths)
        labels = np.append(labels, np.ones(len(image_paths) - len(labels))).astype('int64')
        if mode == 'train':
            image_paths, labels = oversample(image_paths, labels)
        if mode == 'train' and args.add_gan:
            image_paths, labels = gan_aug(image_paths, labels, args.gan_dir)

        self.image_paths = image_paths
        self.labels = np.array(labels)

    def __getitem__(self, index):
        """Take the index of item and returns the image and its labels"""
        image_path = self.image_paths[index]
        img_name = image_path.split('/')[-1]
        img = Image.open(image_path).convert('RGB')
        target = torch.tensor(int(self.labels[index]))

        if self.is_train:
            img = transforms.Resize((500, 500), Image.BILINEAR)(img)
            img = transforms.RandomCrop(self.input_size)(img)
            img = transforms.RandomHorizontalFlip()(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
        else:
            img = transforms.Resize(self.input_size, Image.BILINEAR)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        return img, target, img_name

    def __len__(self):
        return len(self.labels)
