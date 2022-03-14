import logging
import os
import random
import torch
import torchvision.datasets
import numpy as np
import torchvision.transforms as transforms

from torch.backends import cudnn
from yacs.config import CfgNode
from torch.utils.data import DataLoader
from PIL import Image
from complexity_exp.data.dataset import ImageNet
from complexity_exp.data.trigger_dataset import trigger_dataset
from complexity_exp.utils.helper import load_config, denormalize

cfg = load_config()
if torch.cuda.is_available():
    """benchmark设置为True，程序开始时花费额外时间，为整个网络的卷积层搜索最合适的卷积实现算法，
    进而实现网络的加速。要求输入维度和网络（特别是卷积层）结构设置不变，否则程序不断做优化"""
    cudnn.benchmark = True
    if cfg.train.seed is not None:
        np.random.seed(cfg.train.seed)  # Numpy module.
        random.seed(cfg.train.seed)  # Python random module.
        torch.manual_seed(cfg.train.seed)  # Sets the seed for generating random numbers.
        torch.cuda.manual_seed(cfg.train.seed)  # Sets the seed for generating random numbers for the current GPU.
        torch.cuda.manual_seed_all(cfg.train.seed)  # Sets the seed for generating random numbers on all GPUs.
        cudnn.deterministic = True  # 每次返回的卷积算法将是确定的，即默认算法


# 注意：imageNet没有进行normalize
def get_loader(cfg: CfgNode, mode: str, trigger_floder=None):

    dataset_dir = cfg.dataset.dataroot
    dataset_csv = cfg.dataset.dataset_csv
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if mode == 'train':

        tf = transforms.Compose([
            lambda x: Image.open(x).convert('RGB'),
            transforms.Resize((cfg.train.dataloader.resize, cfg.train.dataloader.resize)),
            transforms.RandomHorizontalFlip(),  # 唯一的数据增强
            transforms.ToTensor(),
            normalize  # 注意使用了normalize
        ])
        train_data = ImageNet(dataset_dir, dataset_csv, mode, tf)
        train_loader = DataLoader(
            train_data,
            batch_size=cfg.train.batchsize,
            shuffle=cfg.train.dataloader.shuffle,  # 只有train模式才有shuffle的必要，其他可以作为控制变量
            drop_last=cfg.train.dataloader.drop_last
        )
        return train_loader

    elif mode == 'val':

        tf = transforms.Compose([
            lambda x: Image.open(x).convert('RGB'),
            transforms.Resize((cfg.train.dataloader.resize, cfg.train.dataloader.resize)),
            transforms.ToTensor(),
            normalize  # 注意使用了normalize
        ])

        val_data = ImageNet(dataset_dir, dataset_csv, mode, tf)
        val_loader = DataLoader(
            val_data,
            batch_size=cfg.train.batchsize,
            shuffle=False,
            drop_last=cfg.train.dataloader.drop_last
        )
        return val_loader

    elif mode == 'test':
        tf = transforms.Compose([
            lambda x: Image.open(x).convert('RGB'),
            transforms.Resize((cfg.train.dataloader.resize, cfg.train.dataloader.resize)),
            transforms.ToTensor(),
            normalize  # 注意使用了normalize
        ])

        test_data = ImageNet(dataset_dir, dataset_csv, mode, tf)
        test_loader = DataLoader(
            test_data,
            batch_size=cfg.train.batchsize,
            shuffle=False,
            drop_last=cfg.train.dataloader.drop_last
        )
        return test_loader

    elif mode == 'cover':

        tf = transforms.Compose([
            lambda x: Image.open(x).convert('RGB'),
            transforms.Resize((cfg.train.dataloader.resize, cfg.train.dataloader.resize)),
            transforms.ToTensor(),
            normalize  # 注意使用了normalize
        ])

        cover_set = ImageNet(dataset_dir, dataset_csv, 'val', tf)  # 注意没有cover模式，手动调整为val模式
        cover_loader = DataLoader(
            cover_set,
            batch_size=cfg.watermark.wm_batchsize,
            shuffle=False,
            drop_last=False
        )
        return cover_loader

    elif mode == 'trigger':

        tf = transforms.Compose([
            lambda x: Image.open(x).convert('RGB'),
            transforms.ToTensor(),
            normalize  # 注意使用了normalize
        ])

        trigger_set = trigger_dataset(trigger_floder, tf)
        trigger_loader = DataLoader(
            trigger_set,
            batch_size=cfg.watermark.wm_batchsize,
            shuffle=False,
            drop_last=False
        )
        return trigger_loader


def load_covers(cfg, run_folder):
    device = torch.device(cfg.system.device)
    cover_loader = get_loader(cfg, 'cover')
    cover_imgs, cover_img_labels = [], []

    for wm_idx, (cover_img, cover_img_label) in enumerate(cover_loader):
        cover_img, cover_img_label = cover_img.to(device), cover_img_label.to(device)
        cover_imgs.append(cover_img)
        cover_img_labels.append(cover_img_label)
        if wm_idx == (int(cfg.watermark.wm_num / cfg.watermark.wm_batchsize) - 1):
            break
    # 保存图片
    i = 0
    for img_batch in cover_imgs:
        for img in img_batch:
            torchvision.utils.save_image(denormalize(img),
                                         run_folder + '/original_image/cover_image/' + str(i) + '.png')
            i += 1
    return cover_imgs, cover_img_labels


def load_trigger_labels(cfg, run_folder):
    device = torch.device(cfg.system.device)
    covers_path = run_folder + '/original_image/cover_image/'
    np_labels = np.random.randint(
        cfg.dataset.num_classes,
        size=(int(cfg.watermark.wm_num / cfg.watermark.wm_batchsize), cfg.watermark.wm_batchsize))
    trigger_labels = torch.Tensor(np_labels).type(torch.LongTensor).to(device)

    # 修改cover图片的名称，加上生成的随机标签
    assert trigger_labels.numel() == len(os.listdir(covers_path))  # 验证图片与标签个数是否相等
    covername = os.listdir(covers_path)
    covername.sort(key=lambda x: int(x.split('.')[0]))  # 对文件名按数字排序，固定触发集顺序
    for i, cover in enumerate(covername):
        portion = os.path.splitext(cover)
        os.rename(os.path.join(covers_path, cover), os.path.join(covers_path, str(cover.split('.')[0]) + '_label_' +
                                                                 str(trigger_labels[i // cfg.watermark.wm_batchsize][
                                                                         i % cfg.watermark.wm_batchsize].item()) +
                                                                 portion[1]))
    return trigger_labels  # [batch_num, batchsize]


def load_secret_img(cfg, run_folder):
    global secret_img
    device = torch.device(cfg.system.device)

    transform_logo = transforms.Compose([
        # transforms.Grayscale(num_output_channels=1),
        transforms.Resize((cfg.watermark.wm_resize, cfg.watermark.wm_resize)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 不要遗漏secret_img的归一化，这也是要送入网络训练的
                             std=[0.229, 0.224, 0.225])
    ])

    logo_set = torchvision.datasets.ImageFolder(root=cfg.watermark.logo_root, transform=transform_logo)
    logo_loader = torch.utils.data.DataLoader(logo_set, batch_size=1)

    for _, (logo, __) in enumerate(logo_loader):
        secret_img = logo.expand(cfg.watermark.wm_batchsize, logo.shape[1], logo.shape[2], logo.shape[3]).to(device)
    # 初始化数据时顺便保存图片
    torchvision.utils.save_image(denormalize(secret_img[0]),
                                 run_folder + '/original_image/secret_image/secret_image' + '.png')
    return secret_img
