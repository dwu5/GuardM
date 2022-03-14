import torchvision
import os
from torch.utils.data import Dataset
import glob
import csv
import torch
from complexity_exp.utils.helper import load_config
import random
import numpy as np
from torch.backends import cudnn

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


class ImageNet(Dataset):
    def __init__(self, dataset_dir, dataset_csv, mode, transform: torchvision.transforms):
        super(ImageNet, self).__init__()
        self.dataset_dir = dataset_dir
        self.dataset_csv = dataset_csv
        self.csv_file = os.path.join(dataset_csv, 'mini-ImageNet.csv')
        self.mode = mode
        self.transform = transform
        self.name2label = {}

        for className in sorted(os.listdir(self.dataset_dir)):
            if not os.path.isdir(self.dataset_dir):
                continue
            self.name2label[className] = len(self.name2label.keys())
        self.images, self.labels = self.load_csv()

        # 得到全部数据后，划分数据集
        if mode == 'train':  # 80%
            self.images = self.images[:int(0.01 * len(self.images))]
            self.labels = self.labels[:int(0.01 * len(self.labels))]
        elif mode == 'val':  # 10% = 80% -> 90%
            self.images = self.images[int(0.01 * len(self.images)):int(0.02 * len(self.images))]
            self.labels = self.labels[int(0.01 * len(self.labels)):int(0.02 * len(self.labels))]
        elif mode =='test':  # 10% = 90% -> 100%
            self.images = self.images[int(0.9 * len(self.images)):]
            self.labels = self.labels[int(0.9 * len(self.labels)):]

    def __len__(self):
        return len(self.images)

    # 训练时dataloader的next()方法会反复调用这个方法，获得一个batch的数据
    def __getitem__(self, index):
        img, label = self.images[index], self.labels[index]
        img = self.transform(img)
        label = torch.tensor(label)
        return img, label

    def load_csv(self):
        # 如果已经有了csv文件就不需要重复创建，直接跳到读取
        if not os.path.exists(os.path.join(self.csv_file)):
            images = []
            for name in self.name2label.keys():
                images += glob.glob(os.path.join(self.dataset_dir, name, '*.png'))
                images += glob.glob(os.path.join(self.dataset_dir, name, '*.jpg'))
                images += glob.glob(os.path.join(self.dataset_dir, name, '*.jpeg'))

            random.shuffle(images)  # 关键操作
            # 创建csv文件，并将images中的元素写入csv中
            with open(os.path.join(self.csv_file), mode='w', newline='') as f:
                writer = csv.writer(f)
                for img in images:
                    className = img.split(os.sep)[-2]
                    label = self.name2label[className]
                    writer.writerow([img, label])

        # 下次运行的时候只需要把csv文件加载进来,给self.images和 self.labels重新赋值
        images, labels = [], []
        with open(os.path.join(self.csv_file)) as f:
            reader = csv.reader(f)
            # next(reader)  # 不需要跳过第一行，否则数据集长度不对
            for row in reader:
                img, label = row
                images.append(img)
                label = int(label)
                labels.append(label)
            assert len(images) == len(labels)
            return images, labels


