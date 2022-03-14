from complexity_exp.config.config_node import ConfigNode

config = ConfigNode()

# system
config.system = ConfigNode()
config.system.device = 'cuda'
config.system.GPU = 0
# data
config.dataset = ConfigNode()
config.dataset.dataset_name = 'ImageNet'
config.dataset.dataroot = 'E:\GuardM\complexity_exp\miniImageNetPre'
config.dataset.dataset_csv =  'E:\django-alg-1.0'
config.dataset.num_classes = 100
# watermark
config.watermark = ConfigNode()
config.watermark.logo_root = './data/logo'
config.watermark.ckp_path = './dnn_ckp/'
config.watermark.wm_num = 100
config.watermark.wm_batchsize = 4
config.watermark.wm_resize = 40
# train
config.train = ConfigNode()
config.train.seed = 2022
config.train.fine_tuning = True
config.train.start_epoch = 1
config.train.num_epochs = 100
config.train.batchsize = 8
config.train.lr = 0.001
config.train.momentum = 0.9
config.train.weight_decay = 1e-4
config.train.loss_hyper_param = [3, 5, 1, 0.1]
config.train.print_freq = 10
# train data loader
config.train.dataloader = ConfigNode()
config.train.dataloader.resize = 40
config.train.dataloader.drop_last = True
config.train.dataloader.shuffle = False
config.train.dataloader.pin_memory = False
# runs
config.results = ConfigNode()
config.results.run_folder = 'runs'


def get_default_config():
    return config.clone()
