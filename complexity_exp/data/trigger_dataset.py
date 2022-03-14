import csv
import torchvision
import os
from torch.utils.data import Dataset
import glob
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


class trigger_dataset(Dataset):
    def __init__(self, run_folder, transform: torchvision.transforms):
        super(trigger_dataset, self).__init__()
        self.dataset_dir = os.path.join(run_folder, 'original_image', 'trigger')
        self.dataset_csv = os.path.join(run_folder, 'original_image', 'trigger.csv')
        self.transform = transform
        self.name2label = {}
        self.triggerName = os.listdir(self.dataset_dir)  # 获得每张触发集图片的文件名
        self.triggerName.sort(key=lambda x: int(x.split('_')[0]))  # 按照序号排序
        for name in self.triggerName:
            order = name.split('_')[0]
            self.name2label[order] = int(name.split('_')[2])
        # print("name2label:", self.name2label)  # name2label: {'0': '92', '1': '45',...}
        self.trigger_images, self.trigger_labels = self.load_triggers_csv()

    def __len__(self):
        return len(self.trigger_images)

    def __getitem__(self, index):
        trigger_img, trigger_label = self.trigger_images[index], self.trigger_labels[index]
        trigger_img = self.transform(trigger_img)
        trigger_label = torch.tensor(trigger_label)
        return trigger_img, trigger_label

    def load_triggers_csv(self):
        if not os.path.exists(self.dataset_csv):
            images = []
            for name in sorted(os.listdir(self.dataset_dir)):  # 这里带有随机性。触发集只考虑png类型
                images += os.path.join(self.dataset_dir, name)

            # random.shuffle(images)  # 关键操作
            # 创建csv文件，并将images中的元素写入csv中
            with open(os.path.join(self.dataset_csv), mode='w', newline='') as f:
                writer = csv.writer(f)
                for i, name in enumerate(self.triggerName):
                    trigger_path = os.path.join(self.dataset_dir, name)
                    trigger_label = int(self.name2label[str(i)])
                    writer.writerow([trigger_path, trigger_label])

            # 下次运行的时候只需要把csv文件加载进来,给self.images和 self.labels重新赋值
        trigger_images, trigger_labels = [], []
        with open(os.path.join(self.dataset_csv)) as f:
            reader = csv.reader(f)
            for row in reader:
                img, label = row
                trigger_images.append(img)
                label = int(label)
                trigger_labels.append(label)
            assert len(trigger_images) == len(trigger_labels)
            return trigger_images, trigger_labels
