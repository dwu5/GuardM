import csv
import os
import random
import shutil
import time

import numpy as np
import torch
import logging
import torchvision
import pandas as pd
import seaborn as sn
import torchvision.transforms as transforms
from torch.backends import cudnn
from torch.utils.data import DataLoader
from thop import profile, clever_format
from torchsummary import summary
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from collections import defaultdict
from tensorboardX import SummaryWriter
from complexity_exp.config.defaults import get_default_config
from skimage.metrics import peak_signal_noise_ratio as psnr


def load_config():
    config = get_default_config()
    config.merge_from_file(r"E:\GuardM\complexity_exp\configs\exp.yaml")  # Don't change it to relative path!
    if torch.cuda.is_available():
        config.system.device = 'cuda'
        config.train.dataloader.pin_memory = True
    else:
        config.system.device = 'cpu'
        config.validation.dataloader.pin_memory = False
    config.freeze()
    return config

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


def create_folder(runs_folder):
    if not os.path.exists(runs_folder):
        os.makedirs(runs_folder)

    this_run_folder = os.path.join(runs_folder, f'{time.strftime("%Y.%m.%d--%H-%M-%S")}')

    os.makedirs(this_run_folder)
    os.makedirs(os.path.join(this_run_folder, 'checkpoint/real_acc_criteria'))
    os.makedirs(os.path.join(this_run_folder, 'checkpoint/trigger_acc_criteria'))
    os.makedirs(os.path.join(this_run_folder, 'cat_image/trigger'))
    os.makedirs(os.path.join(this_run_folder, 'cat_image/ext_secret_image'))
    os.makedirs(os.path.join(this_run_folder, 'original_image/cover_image'))
    os.makedirs(os.path.join(this_run_folder, 'original_image/trigger'))
    os.makedirs(os.path.join(this_run_folder, 'original_image/secret_image'))
    os.makedirs(os.path.join(this_run_folder, 'original_image/ext_secret_image'))
    os.makedirs(os.path.join(this_run_folder, 'writer/scalar'))
    os.makedirs(os.path.join(this_run_folder, 'writer/cm/val'))
    os.makedirs(os.path.join(this_run_folder, 'writer/cm/test'))
    # os.makedirs(os.path.join(this_run_folder, 'code'))

    return this_run_folder


def save_code(runs_folder):
    f = open(runs_folder + '/code/wm_main.py', 'w+')
    f = open(runs_folder + '/code/SRNet.py', 'w+')
    f = open(runs_folder + '/code/Google_encoder_new.py', 'w+')
    f = open(runs_folder + '/code/Google_decoder_new.py', 'w+')

    # shutil.copyfile(src, dst):将名为src的文件的内容（无元数据）复制到名为dst的文件中.dst必须是完整的目标文件名
    shutil.copyfile('wm_main.py', runs_folder + "/code/wm_main.py")
    shutil.copyfile('network/Discriminator/SRNet.py', runs_folder + '/code/SRNet.py')
    shutil.copyfile('network/Generator/Google_encoder_new.py', runs_folder + '/code/Google_encoder_new.py')
    shutil.copyfile('network/Decoder/Google_decoder_new.py', runs_folder + '/code/Google_decoder_new.py')

    f.close()


def write_scalars(epoch, file_name, losses_dict, metrics_dict: defaultdict, img_quality_dict, duration):
    with open(file_name, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if epoch == 1:
            if img_quality_dict == None:
                row_to_write = ['epoch'] + [tag for tag in losses_dict.keys()] + [tag for tag in
                                                                                  metrics_dict.keys()] + ['duration']
            else:
                row_to_write = ['epoch'] + [tag for tag in losses_dict.keys()] + [tag for tag in metrics_dict.keys()] + \
                               [tag for tag in img_quality_dict.keys()] + ['duration']
            writer.writerow(row_to_write)
        if img_quality_dict == None:
            row_to_write = [epoch] + ['{:.4f}'.format(value.avg) for value in losses_dict.values()] + \
                           ['{:.6f}'.format(value) for value in metrics_dict.values()] + \
                           ['{:.2f}'.format(duration)]
        else:
            row_to_write = [epoch] + ['{:.4f}'.format(value.avg) for value in losses_dict.values()] + \
                           ['{:.6f}'.format(value) for value in metrics_dict.values()] + \
                           ['{:.2f}'.format(img_quality.avg) for img_quality in img_quality_dict.values()] + \
                           ['{:.2f}'.format(duration)]
        writer.writerow(row_to_write)


def plot_scalars(epoch, run_folder, train_losses_dict: defaultdict, train_metrics_dict,
                                    val_losses_dict: defaultdict, val_metrics_dict, img_quality_dict):
    writer = SummaryWriter(os.path.join(run_folder, 'writer/scalar'))
    # 统一可视化train和val的loss
    for train_tag, val_tag in zip(train_losses_dict.keys(), val_losses_dict.keys()):
        writer.add_scalars(train_tag, {'train': train_losses_dict[train_tag].avg,
                                      'val': val_losses_dict[val_tag].avg}, global_step=epoch)
    # 统一可视化train和val的metrics
    for train_tag, val_tag in zip(train_metrics_dict.keys(), val_metrics_dict.keys()):
        writer.add_scalars(train_tag, {'train': train_metrics_dict[train_tag],
                                       'val': val_metrics_dict[val_tag]}, global_step=epoch)
    # 统一可视化train下的所有loss和metrics, tensorboardX至多在一站图上显示6个曲线，且loss之间没有对比的必要性，索性画在不同的图上
    for loss_tag in train_losses_dict.keys():
        writer.add_scalar('train_scalars/'+loss_tag, train_losses_dict[loss_tag].avg, global_step=epoch)
    for metirc_tag in train_metrics_dict.keys():
        writer.add_scalars('train_scalars/'+metirc_tag, {metirc_tag: train_metrics_dict[metirc_tag]}, global_step=epoch)
    # 统一可视化val下的所有loss和metrics
    for loss_tag in val_losses_dict.keys():
        writer.add_scalar('val_scalars/' + loss_tag, val_losses_dict[loss_tag].avg, global_step=epoch)
    for metirc_tag in val_metrics_dict.keys():
        writer.add_scalars('val_scalars/' + metirc_tag, {metirc_tag: val_metrics_dict[metirc_tag]}, global_step=epoch)
    # 统一可视化val中的psnr和ssim
    for img_tag in img_quality_dict.keys():
        writer.add_scalar('val_scalars/' + img_tag, img_quality_dict[img_tag].avg, global_step=epoch)
    writer.close()


def plot_confusion_matrix(epoch, run_folder, tag, y_pred, y_true, data_loader: DataLoader):
    if "test" in tag:
        writer = SummaryWriter(os.path.join(run_folder, 'writer/cm/test'))
    else:
        writer = SummaryWriter(os.path.join(run_folder, 'writer/cm/val'))
    # constant for classes
    if tag == 'train_dis' or tag == 'val_dis' or tag == 'test_dis':
        classes = [0, 1]  # 鉴别器的判断是否为隐写图片的标签
        fig_size = 5
    else:
        classes = list(data_loader.dataset.name2label.values())
        fig_size = 20  # 最佳尺寸
    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred, labels=classes)
    df_cm = pd.DataFrame(cf_matrix, index=[i for i in classes], columns=[i for i in classes])  # 直接显示正确预测到特定类别的个数，不是百分比
    plt.figure(figsize=(fig_size+5, fig_size))  # 可以调整混淆矩阵热力图的宽和高，方便查看
    # 设置annot=True后，图上已经有数值，因此暂时不对混淆矩阵本身作保存
    writer.add_figure(tag + "_confusion_matrix", sn.heatmap(df_cm, annot=True).get_figure(), global_step=epoch)
    writer.close()


def measure_complexity(epoch: int, run_folder: str, model: torch.nn.Module, input: torch.tensor):
    path = os.path.join(run_folder, 'complexity_info.log')
    global start_time
    with open(path, 'a+') as f:
        f.write("\n==>epoch:{}\n".format(epoch))
        f.write("\n##################### Using numel #####################\n")
        f.write("Total paramerters:{}\n".format(sum(x.numel() for x in model.parameters())))

        f.write("\n##################### Using thop #####################\n")
        # 统计所有batch的FLOPs，而不是求平均的单个batch
        flops, params = profile(model, inputs=(input,), f=f)
        flops_cf, params_cf = clever_format([flops, params], '%.3f')
        f.write("FLOPs:{}({}) params:{}({})\n".format(flops_cf, flops, params_cf, params))

        f.write("\n##################### Using torchsummary #####################\n")
        # 只和参数相关，Input size、Forward/backward pass size (MB)都与batch成正比，因此最终的Estimated Total Size也会改变
        # 但Params size (MB)不会随着batch多少而改变
        summary(model, f=f, input_size=(input.size()[1], input.size()[2], input.size()[3]), batch_size=input.size()[0])

        # f.write("\n##################### Using stat #####################")
        # # 所有指标都不会随着batch多少而改变
        # stat(model, input_size=(input.size()[1], input.size()[2], input.size()[3]), f=f)

        # f.write("\n##################### Using ptflops #####################")
        # # flops和params都不随输入而改变
        # flops, params = get_model_complexity_info(model, (input.size()[1], input.size()[2], input.size()[3]),
        #                                           as_strings=True, print_per_layer_stat=True, ost=f)
        # f.write("FLOPs:{} params:{}".format(flops, params))


def save_cat_image(cfg, epoch, run_folder, cover_img, trigger, secret_img, trigger_ext_output):
    # denormalize
    cover_img, trigger, secret_img, trigger_ext_output = \
        denormalize(cover_img), denormalize(trigger), denormalize(secret_img), denormalize(trigger_ext_output)
    # 保存雪碧图
    # 如果是灰度水印图像，那么通道不一样, 做concatenate的时候维度不匹配，需要分开两次保存
    result_ste_img = torch.cat([cover_img, trigger], 0)
    torchvision.utils.save_image(result_ste_img,
                                 run_folder + '/cat_image/trigger/Epoch_' + str(epoch) + '.png',
                                 nrow=cfg.watermark.wm_batchsize,
                                 padding=1, normalize=False)  # 试探：False
    result_sec_img = torch.cat([secret_img, trigger_ext_output], 0)
    torchvision.utils.save_image(result_sec_img,
                                 run_folder + '/cat_image/ext_secret_image/Epoch_' + str(epoch) + '.png',
                                 nrow=cfg.watermark.wm_batchsize,
                                 padding=1, normalize=False)  # 试探：False


def save_separate_image(epoch, run_folder, triggers, trigger_labels, trigger_ext_output):
    # 根据wm_acc单独保存图片
    # 首先还原图片
    triggers, trigger_ext_output = denormalize(triggers), denormalize(trigger_ext_output)
    # 清空文件夹再追加图片
    trigger_root = run_folder + '/original_image/trigger/'
    ex_root = run_folder + '/original_image/ext_secret_image/'

    if os.path.getsize(trigger_root) > 0:
        shutil.rmtree(trigger_root)
        os.makedirs(trigger_root)

    if os.path.getsize(ex_root) > 0:
        shutil.rmtree(ex_root)
        os.makedirs(ex_root)

    for i, img in enumerate(triggers):
        torchvision.utils.save_image(img,  trigger_root + str(i) + '_label_' +
            str(trigger_labels[i//cfg.watermark.wm_batchsize][i % cfg.watermark.wm_batchsize].item()) + '_epoch_' + str(epoch) + '.png')
    for i, img in enumerate(trigger_ext_output):
        torchvision.utils.save_image(img, ex_root + 'epoch_' + str(
            epoch) + '_ext_wm' + str(i) + '.png')


def save_checkpoint(epoch, run_folder, Hidnet, Disnet, Extnet, Dnnet, optimizerH, optimizerD, optimizerE, optimizerN,
                    val_losses_dict: defaultdict, val_metrics_dict, img_quality_dict, criteria, best):
    checkpoint_folder = os.path.join(run_folder, 'checkpoint')
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)

    Hstate = {
        'Hidnet': Hidnet.state_dict(),
        'optimizerH': optimizerH.state_dict(),
        'loss_H': val_losses_dict['loss_H'].avg,
        'loss_mse': val_losses_dict['loss_mse'].avg,
        'loss_ssim': val_losses_dict['loss_ssim'].avg,
        'loss_adv': val_losses_dict['loss_adv'].avg,
        'loss_dnn': val_losses_dict['loss_dnn'].avg,
        'psnr': img_quality_dict['psnr'].avg,
        'epoch': epoch
    }
    Dstate = {
        'Disnet': Disnet.state_dict(),
        'optimizerD': optimizerD.state_dict(),
        'loss_Dis': val_losses_dict['loss_Dis'].avg,
        'epoch': epoch
    }
    Estate = {
        'Extnet': Extnet.state_dict(),
        'optimizerE': optimizerE.state_dict(),
        'loss_E': val_losses_dict['loss_DEC'].avg,
        'epoch': epoch
    }
    Nstate = {
        'Dnnet': Dnnet.state_dict(),
        'optimizerN': optimizerN.state_dict(),
        'loss_cat_Dnn': val_losses_dict['loss_cat_Dnn'].avg,
        'real_acc': val_metrics_dict['real_acc'],
        'trigger_acc': val_metrics_dict['trigger_acc'],
        'cover_acc': val_metrics_dict['cover_acc'],
        'precision': val_metrics_dict['precision'],
        'recall': val_metrics_dict['recall'],
        'f1': val_metrics_dict['f1'],
        'epoch': epoch
    }
    save_path = os.path.join(run_folder, 'checkpoint', criteria)
    # 先清空保存checkpoint的文件夹
    shutil.rmtree(save_path)
    os.makedirs(save_path)
    torch.save(Hstate, save_path + '/Hidnet_' + str(epoch) + 'epoch_' + '_' + str(round(best.item(), 6)) + '.pt')
    torch.save(Dstate, save_path + '/Disnet_' + str(epoch) + 'epoch_' + '_' + str(round(best.item(), 6)) + '.pt')
    torch.save(Estate, save_path + '/Extnet_' + str(epoch) + 'epoch_' + '_' + str(round(best.item(), 6)) + '.pt')
    torch.save(Nstate, save_path + '/Dnnet_' + str(epoch) + 'epoch_' + '_' + str(round(best.item(), 6)) + '.pt')

    logging.info('Save checkpoint to {}'.format(checkpoint_folder))

def cal_psnr(cover_imgs, triggers):

    cover_imgs, triggers = denormalize(cover_imgs), denormalize(triggers)
    wm_batchsize = triggers.size(0)

    gary_tf = transforms.Grayscale(num_output_channels=1)
    cover_imgs = torch.as_tensor(gary_tf(cover_imgs)).cpu().numpy()
    triggers = torch.as_tensor(gary_tf(triggers)).cpu().numpy()
    total_psnr = 0

    for cover_img, trigger in zip(cover_imgs, triggers):
        total_psnr += psnr(cover_img, trigger)

    return total_psnr / wm_batchsize


def RGB_to_gray(img: torch.tensor):
    tf = transforms.Grayscale(num_output_channels=1)
    return tf(img)


def denormalize(img_hat):
    """
    对ImageNet数据集通用的mean和std，将数据标准化到（-1,1）因此在可视化时需要denormalize
    :param x_hat:= (x-mean)/std  => x = x_hat*std + mean
    :return:
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
    std = torch.tensor(std).unsqueeze(1).unsqueeze(1)
    img = img_hat.cpu() * std +mean

    return img
