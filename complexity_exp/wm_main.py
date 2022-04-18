from __future__ import print_function

import logging
import sys
import warnings
import torch.nn as nn
import torch.optim as optim
from itertools import cycle
from torchmetrics import Accuracy, Precision, Recall, F1

from complexity_exp.data.load_data import *
from complexity_exp.models.resnet import resnet18
from complexity_exp.network.Decoder.Google_decoder_new import Google_decoderNet
from complexity_exp.network.Discriminator.SRNet import Srnet
from complexity_exp.network.Generator.Google_encoder_new import Google_encoderNet
from complexity_exp.utils.AverageMeter import AverageMeter
from complexity_exp.utils.SSIM import SSIM
from complexity_exp.utils.helper import *

warnings.filterwarnings("ignore")
# 实验配置准备
cfg = load_config()
os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.system.GPU)
device = torch.device(cfg.system.device)
run_folder = create_folder(cfg.results.run_folder)
# save_code(run_folder)  # 在用Django框架时有bug
# 初始化日志打印，输出本次实验的超参配置
logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[
    logging.FileHandler(os.path.join(run_folder, f'run.log')), logging.StreamHandler(sys.stdout)])
logging.info("Experiment Configuration:")
logging.info(cfg)
# 使用cudnn库加速，并设置随机种子使实验结果可复现
if torch.cuda.is_available():
    cudnn.benchmark = True
    if cfg.train.seed is not None:
        np.random.seed(cfg.train.seed)  # Numpy module.
        random.seed(cfg.train.seed)  # Python random module.
        torch.manual_seed(cfg.train.seed)  # Sets the seed for generating random numbers.
        torch.cuda.manual_seed(cfg.train.seed)  # Sets the seed for generating random numbers for the current GPU.
        torch.cuda.manual_seed_all(cfg.train.seed)  # Sets the seed for generating random numbers on all GPUs.
        cudnn.deterministic = True  # 将这个flag置为True，每次返回的卷积算法将是确定的，即默认算法

        warnings.warn('You have choosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
        logging.info('torch.cuda is available!')
# 加载数据：训练集、验证集、测试集、载体图像及其标签、触发集标签、秘密图像
train_loader, val_loader, test_loader = get_loader(cfg, 'train'), get_loader(cfg, 'val'), get_loader(cfg, 'test')
covers = load_covers(cfg, run_folder)
cover_imgs, cover_img_labels = covers[0], covers[1]
trigger_labels, secret_img = load_trigger_labels(cfg, run_folder), load_secret_img(cfg, run_folder)
logging.info("train_loader:{} val_loader:{} test_loader:{}".format(len(train_loader.dataset), len(val_loader.dataset),
                                                                   len(test_loader.dataset)))
# 定义网络
Hidnet = Google_encoderNet()
Extnet = Google_decoderNet()
Disnet = Srnet()
Dnnet = resnet18()
# 将模型转移到cuda上实现数据并行
Hidnet = nn.DataParallel(Hidnet.to(device))
Extnet = nn.DataParallel(Extnet.to(device))
Disnet = nn.DataParallel(Disnet.to(device))
Dnnet = nn.DataParallel(Dnnet.to(device))
# 定义各网络的损失函数和优化器
criterionH_mse = nn.MSELoss()
criterionH_ssim = SSIM()
optimizerH = optim.Adam(Hidnet.parameters(), lr=cfg.train.lr, betas=(0.5, 0.999))

criterionD = nn.CrossEntropyLoss()
optimizerD = optim.Adam(Disnet.parameters(), lr=cfg.train.lr, betas=(0.5, 0.999))

criterionN = nn.CrossEntropyLoss()
optimizerN = optim.Adam(Dnnet.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

criterionE = nn.MSELoss()
optimizerE = optim.Adam(Extnet.parameters(), lr=cfg.train.lr)

# 定义对抗训练的真实标签
valid = torch.LongTensor(cfg.watermark.wm_batchsize, 1).fill_(1.0).view(-1, 1).to(
    device)  # tensor([[1.], [1.], ..., [1.]])
fake = torch.LongTensor(cfg.watermark.wm_batchsize, 1).fill_(0.0).view(-1, 1).to(
    device)  # tensor([[0.], [0.], ..., [0.]])
valid = torch.squeeze(valid)
fake = torch.squeeze(fake)


def train(epoch):
    logging.info('Training epoch: %d' % epoch)
    # 模型开启train模式
    Dnnet.train()
    Hidnet.train()
    Extnet.train()
    Disnet.train()

    step = 1
    epoch_duration = 0
    real_preds, real_trues, trigger_preds, trigger_trues, cover_preds, cover_trues, dis_preds, dis_trues = [], [], [], [], [], [], [], []
    # 用字典保存每个epoch的指标值
    losses_dict = defaultdict(AverageMeter)  # 所有loss
    metrics_dict = defaultdict()  # 宿主模型的正向指标
    # 加水印后宿主模型的正向指标
    real_acc = Accuracy(num_classes=cfg.dataset.num_classes, average='weighted')
    trigger_acc = Accuracy(num_classes=cfg.dataset.num_classes, average='weighted')
    cover_acc = Accuracy(num_classes=cfg.dataset.num_classes, average='weighted')
    dis_acc = Accuracy(num_classes=2, average='micro')  # micro就是把所有类（这里就只有两个类）汇总在一起后计算acc
    precision = Precision(num_classes=cfg.dataset.num_classes, average='weighted')
    recall = Recall(num_classes=cfg.dataset.num_classes, average='weighted')
    f1 = F1(num_classes=cfg.dataset.num_classes, average='weighted')

    for batch_idx, (input, label) in enumerate(train_loader):
        # 加载每批batch的相关训练数据
        input, label = input.to(device), label.to(device)  # transforms.Normalize之前： 0-1,torch.float32, torch.int64
        cover_img = cover_imgs[batch_idx % len(cover_imgs)].to(device)
        cover_img_label = cover_img_labels[batch_idx % len(cover_imgs)].to(device)
        trigger_label = trigger_labels[batch_idx % len(cover_imgs)].to(device)
        """############################### Disnet ###############################"""
        optimizerD.zero_grad()
        trigger = Hidnet(cover_img, secret_img)
        fake_dis_output = Disnet(trigger.detach())  # train on fake data
        real_dis_output = Disnet(cover_img)  # train on real data
        loss_D_fake = criterionD(fake_dis_output, fake)
        loss_D_real = criterionD(real_dis_output, valid)
        loss_Dis = loss_D_fake + loss_D_real
        loss_Dis.backward()
        optimizerD.step()

        """############################### Hidnet and Extnet ###############################"""
        optimizerH.zero_grad()
        optimizerD.zero_grad()
        optimizerE.zero_grad()
        optimizerN.zero_grad()

        trigger_dis_output = Disnet(trigger)  # 输出判断预测结果
        trigger_dnn_output = Dnnet(trigger)  # 输出分类预测结果
        trigger_ext_output = Extnet(trigger)  # 输出提取出的水印张量
        # 先计算编码器的各种loss
        loss_mse = criterionH_mse(cover_img, trigger)
        loss_ssim = criterionH_ssim(cover_img, trigger)  # 这里返回的直接是SSIM值，不是损失值
        loss_adv = criterionD(trigger_dis_output, valid)
        loss_dnn = criterionN(trigger_dnn_output, trigger_label)
        loss_H = cfg.train.loss_hyper_param[0] * loss_mse + cfg.train.loss_hyper_param[1] * (1 - loss_ssim) + \
                 cfg.train.loss_hyper_param[2] * loss_adv + cfg.train.loss_hyper_param[3] * loss_dnn
        loss_E = criterionE(trigger_ext_output, secret_img)

        loss_E.backward(retain_graph=True)
        optimizerE.step()

        loss_H.backward()
        optimizerH.step()

        """############################### Dnnet ###############################"""
        # dnn训练计时开始
        dnn_start_time = time.time()
        # 合并输入
        inputs = torch.cat([input, trigger.detach()], dim=0)
        labels = torch.cat([label, trigger_label], dim=0)
        # 放入宿主模型训练（加入了关于载体图像无效性的验证）
        optimizerN.zero_grad()
        dnn_cat_output = Dnnet(
            inputs)  # e.g. tensor([[-0.4108, ..., -0.3460], ..., [0.1289, ..., -0.3018 ]], shape: [wm_batchsize+batchsize, 100]
        cover_output = Dnnet(cover_img)  # 测试宿主模型是否能识别载体图像(否则验证水印时无效)
        loss_cat_Dnn = criterionN(dnn_cat_output, labels)
        loss_cat_Dnn.backward()
        optimizerN.step()
        # dnn训练计时结束并累加
        epoch_duration += time.time() - dnn_start_time
        # 分离输出 通过argmax()返回指定维度上最大值下标索引，转换成预测标签
        real_pred = dnn_cat_output[0:cfg.train.batchsize].argmax(dim=1)
        trigger_pred = dnn_cat_output[cfg.train.batchsize:].argmax(dim=1)
        cover_pred = cover_output.argmax(dim=1)
        fake_dis_pred = torch.max(fake_dis_output, dim=1)[1]  # [a, b]返回最大值的索引值
        real_dis_true = torch.max(real_dis_output, dim=1)[1]
        # 分别存储正常数据集和触发集的累计输出,由于是list类型，需要转换成numpy
        real_preds.extend(real_pred.cpu().numpy())
        real_trues.extend(label.cpu().numpy())
        trigger_preds.extend(trigger_pred.cpu().numpy())
        trigger_trues.extend(trigger_label.cpu().numpy())
        cover_preds.extend(cover_pred.cpu().numpy())
        cover_trues.extend(cover_img_label.cpu().numpy())
        dis_preds.extend(torch.cat([fake_dis_pred, real_dis_true], dim=0).cpu().numpy())
        dis_trues.extend(torch.cat([fake, valid], dim=0).cpu().numpy())
        # 利用TorchMetrics更新宿主模型的六个正向指标
        real_acc.update(real_pred.cpu(), label.cpu())
        trigger_acc.update(trigger_pred.cpu(), trigger_label.cpu())
        cover_acc.update(cover_pred.cpu(), cover_img_label.cpu())
        dis_acc.update(torch.cat([fake_dis_pred, real_dis_true], dim=0).cpu(), torch.cat([fake, valid], dim=0).cpu())
        precision.update(real_pred.cpu(), label.cpu())
        recall.update(real_pred.cpu(), label.cpu())
        f1.update(real_pred.cpu(), label.cpu())
        # 只是将loss的值保存暂存在字典中，方便后续用for循环进行update操作
        temp_losses_dict = {
            'loss_Dis': loss_Dis.item(),
            'loss_mse': loss_mse.item(),
            'loss_ssim': loss_ssim.item(),
            'loss_adv': loss_adv.item(),
            'loss_dnn': loss_dnn.item(),
            'loss_H': loss_H.item(),
            'loss_E': loss_E.item(),
            'loss_cat_Dnn': loss_cat_Dnn.item()
        }
        for tag, metric in temp_losses_dict.items():
            if tag == 'loss_cat_Dnn':
                losses_dict[tag].update(metric, inputs.size(0))  # 注意分母, loss_cat_Dnn是合并输入的维度
            else:
                losses_dict[tag].update(metric, trigger.size(0))  # 其他网络模块的损失函数，都是其输入(trigger)的维度
        # 打印输出
        if step % cfg.train.print_freq == 0 or step == (len(train_loader)):
            logging.info(
                '[{}/{}][{}/{}] Loss Dis: {:.4f} Loss_H: {:.4f} (mse: {:.4f} ssim: {:.4f} adv: {:.4f} dnn: {:.4f}) '
                'Loss_E: {:.4f} Loss_cat_Dnn: {:.4f}'.format(
                    epoch, cfg.train.num_epochs, step, len(train_loader),
                    losses_dict['loss_Dis'].avg, losses_dict['loss_H'].avg, losses_dict['loss_mse'].avg,
                    losses_dict['loss_ssim'].avg, losses_dict['loss_adv'].avg, losses_dict['loss_dnn'].avg,
                    losses_dict['loss_E'].avg, losses_dict['loss_cat_Dnn'].avg))
            logging.info("\t\t\t\tReal acc: {:.4%} Trigger acc: {:.4%} Cover acc: {:.4%} Dis acc: {:.4%} "
                         "Precision: {:.4%} Recall: {:.4%} F1: {:.4%}".format(
                real_acc.compute(), trigger_acc.compute(), cover_acc.compute(), dis_acc.compute(), precision.compute(),
                recall.compute(), f1.compute()))
            logging.info('-' * 160)
        step += 1
    logging.info('Epoch {} training duration {:.2f} sec'.format(epoch, epoch_duration))
    logging.info('-' * 160)

    metrics_dict['real_acc'] = real_acc.compute()
    metrics_dict['trigger_acc'] = trigger_acc.compute()
    metrics_dict['cover_acc'] = cover_acc.compute()
    metrics_dict['dis_acc'] = dis_acc.compute()
    metrics_dict['precision'] = precision.compute()
    metrics_dict['recall'] = recall.compute()
    metrics_dict['f1'] = f1.compute()

    write_scalars(epoch, os.path.join(run_folder, 'train.csv'), losses_dict, metrics_dict, None, epoch_duration)
    plot_confusion_matrix(epoch, run_folder, 'train_real', real_preds, real_trues, train_loader)
    plot_confusion_matrix(epoch, run_folder, 'train_trigger', trigger_preds, trigger_trues, train_loader)
    plot_confusion_matrix(epoch, run_folder, 'train_dis', dis_preds, dis_trues, None)

    return losses_dict, metrics_dict


def validation(epoch):
    # 模型开启评估模式
    Dnnet.eval()
    Hidnet.eval()
    Extnet.eval()
    Disnet.eval()

    epoch_duration = 0
    real_preds, real_trues, trigger_preds, trigger_trues, cover_preds, cover_trues, dis_preds, dis_trues = [], [], [], [], [], [], [], []
    triggers = torch.Tensor()  # 创建空张量，存放所有trigger
    # 用字典保存每个epoch的指标值
    losses_dict = defaultdict(AverageMeter)  # 所有loss
    metrics_dict = defaultdict()  # 宿主模型的正向指标
    img_quality_dict = defaultdict(AverageMeter)  # PSNR和SSIM
    # 加水印后宿主模型的正向指标
    real_acc = Accuracy(num_classes=cfg.dataset.num_classes, average='weighted')
    trigger_acc = Accuracy(num_classes=cfg.dataset.num_classes, average='weighted')
    cover_acc = Accuracy(num_classes=cfg.dataset.num_classes, average='weighted')
    dis_acc = Accuracy(num_classes=2, average='micro')  # 鉴别器的准确率，理想情况希望是50%左右
    precision = Precision(num_classes=cfg.dataset.num_classes, average='weighted')
    recall = Recall(num_classes=cfg.dataset.num_classes, average='weighted')
    f1 = F1(num_classes=cfg.dataset.num_classes, average='weighted')

    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(val_loader):
            # 加载每批batch的相关训练数据
            input, label = input.to(device), label.to(device)
            cover_img = cover_imgs[batch_idx % len(cover_imgs)].to(device)
            cover_img_label = cover_img_labels[batch_idx % len(cover_img)].to(device)
            trigger_label = trigger_labels[batch_idx % len(cover_imgs)].to(device)
            """############################### Disnet ###############################"""
            trigger = Hidnet(cover_img, secret_img)
            # 最后一次遍历触发集时，保存所有触发集
            if batch_idx >= len(val_loader.dataset) / cfg.train.batchsize - len(cover_imgs):
                triggers = torch.cat([triggers, trigger.detach().cpu()], dim=0)
            fake_dis_output = Disnet(trigger.detach())
            real_dis_output = Disnet(cover_img)
            loss_D_fake = criterionD(fake_dis_output, fake)
            loss_D_real = criterionD(real_dis_output, valid)
            loss_Dis = loss_D_fake + loss_D_real
            """############################### Hidnet and Extnet ###############################"""
            trigger_dis_output = Disnet(trigger)  # 输出判断预测结果
            trigger_dnn_output = Dnnet(trigger)  # 输出分类预测结果
            trigger_ext_output = Extnet(trigger)  # 输出提取出的水印张量
            # 先计算编码器的各种loss
            loss_mse = criterionH_mse(cover_img, trigger)
            loss_ssim = criterionH_ssim(cover_img, trigger)
            loss_adv = criterionD(trigger_dis_output, valid)
            loss_dnn = criterionN(trigger_dnn_output, trigger_label)
            loss_H = cfg.train.loss_hyper_param[0] * loss_mse + cfg.train.loss_hyper_param[1] * (1 - loss_ssim) + \
                     cfg.train.loss_hyper_param[2] * loss_adv + cfg.train.loss_hyper_param[3] * loss_dnn
            loss_E = criterionE(trigger_ext_output, secret_img)

            """############################### Dnnet ###############################"""
            # dnn前向传播计时开始
            dnn_start_time = time.time()
            # 合并输入
            inputs = torch.cat([input, trigger.detach()], dim=0)
            labels = torch.cat([label, trigger_label], dim=0)
            # 前向传播
            dnn_cat_output = Dnnet(inputs)
            cover_output = Dnnet(cover_img)  # 测试宿主模型是否能识别载体图像(否则验证水印时无效)
            loss_cat_Dnn = criterionN(dnn_cat_output, labels)
            # dnn前向传播计时结束并累加
            epoch_duration += time.time() - dnn_start_time
            # 分离输出 通过argmax()返回指定维度上最大值下标索引，转换成预测标签
            real_pred = dnn_cat_output[0:cfg.train.batchsize].argmax(dim=1)
            trigger_pred = dnn_cat_output[cfg.train.batchsize:].argmax(dim=1)
            cover_pred = cover_output.argmax(dim=1)
            fake_dis_pred = torch.max(fake_dis_output, dim=1)[1]
            real_dis_true = torch.max(real_dis_output, dim=1)[1]
            # 分别存储正常数据集和触发集的累计输出,由于是list类型，需要转换成numpy
            real_preds.extend(real_pred.cpu().numpy())
            real_trues.extend(label.cpu().numpy())
            trigger_preds.extend(trigger_pred.cpu().numpy())
            trigger_trues.extend(trigger_label.cpu().numpy())
            cover_preds.extend(cover_pred.cpu().numpy())
            cover_trues.extend(cover_img_label.cpu().numpy())
            dis_preds.extend(torch.cat([fake_dis_pred, real_dis_true], dim=0).cpu().numpy())
            dis_trues.extend(torch.cat([fake, valid], dim=0).cpu().numpy())
            # 利用TorchMetrics更新宿主模型的六个正向指标
            real_acc.update(real_pred.cpu(), label.cpu())
            trigger_acc.update(trigger_pred.cpu(), trigger_label.cpu())
            cover_acc.update(cover_pred.cpu(), cover_img_label.cpu())
            dis_acc.update(torch.cat([fake_dis_pred, real_dis_true], dim=0).cpu(),
                           torch.cat([fake, valid], dim=0).cpu())
            precision.update(real_pred.cpu(), label.cpu())
            recall.update(real_pred.cpu(), label.cpu())
            f1.update(real_pred.cpu(), label.cpu())
            # 只是将loss的值保存暂存在字典中，方便后续用for循环进行update操作
            temp_losses_dict = {
                'loss_Dis': loss_Dis.item(),
                'loss_mse': loss_mse.item(),
                'loss_ssim': loss_ssim.item(),
                'loss_adv': loss_adv.item(),
                'loss_dnn': loss_dnn.item(),
                'loss_H': loss_H.item(),
                'loss_E': loss_E.item(),
                'loss_cat_Dnn': loss_cat_Dnn.item()
            }
            for tag, metric in temp_losses_dict.items():
                if tag == 'loss_cat_Dnn':
                    losses_dict[tag].update(metric, inputs.size(0))  # 注意分母, loss_cat_Dnn是合并输入的维度
                else:
                    losses_dict[tag].update(metric, trigger.size(0))  # 其他网络模块的损失函数，都是其输入(trigger)的维度
            # 更新PSNR
            batch_psnr = cal_psnr(cover_img.detach(), trigger.detach())
            img_quality_dict['psnr'].update(batch_psnr, trigger.size(0))

    logging.info(
        '[{}/{}] Loss Dis: {:.4f} Loss_H: {:.4f} (mse: {:.4f} ssim: {:.4f} adv: {:.4f} dnn: {:.4f}) '
        'Loss_E: {:.4f} Loss_cat_Dnn: {:.4f}'.format(
            epoch, cfg.train.num_epochs,
            losses_dict['loss_Dis'].avg, losses_dict['loss_H'].avg, losses_dict['loss_mse'].avg,
            losses_dict['loss_ssim'].avg, losses_dict['loss_adv'].avg, losses_dict['loss_dnn'].avg,
            losses_dict['loss_E'].avg, losses_dict['loss_cat_Dnn'].avg))
    logging.info("\t\t\t\tReal acc: {:.4%} Trigger acc: {:.4%} Cover acc: {:.4%} Dis acc: {:.4%} "
                 "Precision: {:.4%} Recall: {:.4%} F1: {:.4%} PSNR: {:.2f}".format(
        real_acc.compute(), trigger_acc.compute(), cover_acc.compute(), dis_acc.compute(), precision.compute(),
        recall.compute(),
        f1.compute(), batch_psnr))
    logging.info('Epoch {} validation duration {:.2f} sec'.format(epoch, epoch_duration))
    logging.info('-' * 160)

    metrics_dict['real_acc'] = real_acc.compute()
    metrics_dict['trigger_acc'] = trigger_acc.compute()
    metrics_dict['cover_acc'] = cover_acc.compute()
    metrics_dict['dis_acc'] = dis_acc.compute()
    metrics_dict['precision'] = precision.compute()
    metrics_dict['recall'] = recall.compute()
    metrics_dict['f1'] = f1.compute()

    write_scalars(epoch, os.path.join(run_folder, 'val.csv'), losses_dict, metrics_dict, img_quality_dict,
                  epoch_duration)
    plot_confusion_matrix(epoch, run_folder, 'val_real', real_preds, real_trues, val_loader)
    plot_confusion_matrix(epoch, run_folder, 'val_trigger', trigger_preds, trigger_trues, val_loader)
    plot_confusion_matrix(epoch, run_folder, 'val_dis', dis_preds, dis_trues, None)
    save_cat_image(cfg, epoch, run_folder, cover_img, trigger, secret_img, trigger_ext_output)

    return losses_dict, metrics_dict, img_quality_dict, triggers, trigger_labels, trigger_ext_output


def test(run_folder, trigger_floder):
    # 不需要隐写模型的参与
    Disnet.eval()
    Dnnet.eval()

    test_duration = 0
    real_preds, real_trues, trigger_preds, trigger_trues, cover_preds, cover_trues, dis_preds, dis_trues = [], [], [], [], [], [], [], []
    # 保存损失的字典，在每个batch中更新平均损失
    losses_dict = defaultdict(AverageMeter)
    # 保存图像质量指标的字典
    img_quality_dict = defaultdict(AverageMeter)
    # 加水印后宿主模型的正向指标
    real_acc = Accuracy(num_classes=cfg.dataset.num_classes, average='weighted')
    trigger_acc = Accuracy(num_classes=cfg.dataset.num_classes, average='weighted')
    cover_acc = Accuracy(num_classes=cfg.dataset.num_classes, average='weighted')
    dis_acc = Accuracy(num_classes=2, average='micro')  # 鉴别器的准确率，理想情况希望是50%左右
    precision = Precision(num_classes=cfg.dataset.num_classes, average='weighted')
    recall = Recall(num_classes=cfg.dataset.num_classes, average='weighted')
    f1 = F1(num_classes=cfg.dataset.num_classes, average='weighted')
    # 加载需要测试的触发集
    trigger_loader = get_loader(cfg, 'trigger', trigger_floder=trigger_floder)

    with torch.no_grad():
        # 这里使用了cycle对触发集中的数据循环测试
        for batch_idx, ((input, label), (trigger, trigger_label)) in enumerate(zip(test_loader, cycle(trigger_loader))):
            input, label, trigger, trigger_label = input.to(device), label.to(device), trigger.to(
                device), trigger_label.to(device)
            cover_img = cover_imgs[batch_idx % len(cover_imgs)].to(device)
            cover_img_label = cover_img_labels[batch_idx % len(cover_imgs)].to(device)
            """############################### Disnet ###############################"""
            # 直接从文件中读取触发集，不需要通过Hidnet临时生成
            fake_dis_output = Disnet(trigger.detach())
            real_dis_output = Disnet(cover_img)
            loss_D_fake = criterionD(fake_dis_output, fake)
            loss_D_real = criterionD(real_dis_output, valid)
            loss_Dis = loss_D_fake + loss_D_real
            """############################### Dnnet ###############################"""
            # dnn前向传播计时开始
            dnn_start_time = time.time()
            # 合并输入
            inputs = torch.cat([input, trigger.detach()], dim=0)
            labels = torch.cat([label, trigger_label], dim=0)
            # 前向推理
            dnn_cat_output = Dnnet(inputs)
            cover_output = Dnnet(cover_img)  # 测试宿主模型是否能识别载体图像(否则验证水印时无效)
            loss_cat_Dnn = criterionN(dnn_cat_output, labels)
            # dnn前向推理计时结束并累加
            test_duration += time.time() - dnn_start_time
            # 分离输出 通过argmax()返回指定维度上最大值下标索引，转换成预测标签
            real_pred = dnn_cat_output[0:cfg.train.batchsize].argmax(dim=1)
            trigger_pred = dnn_cat_output[cfg.train.batchsize:].argmax(dim=1)
            cover_pred = cover_output.argmax(dim=1)
            fake_dis_pred = torch.max(fake_dis_output, dim=1)[1]
            real_dis_true = torch.max(real_dis_output, dim=1)[1]
            # 分别存储正常数据集和触发集的累计输出,由于是list类型，需要转换成numpy
            real_preds.extend(real_pred.cpu().numpy())
            real_trues.extend(label.cpu().numpy())
            trigger_preds.extend(trigger_pred.cpu().numpy())
            trigger_trues.extend(trigger_label.cpu().numpy())
            cover_preds.extend(cover_pred.cpu().numpy())
            cover_trues.extend(cover_img_label.cpu().numpy())
            dis_preds.extend(torch.cat([fake_dis_pred, real_dis_true], dim=0).cpu().numpy())
            dis_trues.extend(torch.cat([fake, valid], dim=0).cpu().numpy())
            # 利用TorchMetrics更新宿主模型的六个正向指标
            real_acc.update(real_pred.cpu(), label.cpu())
            trigger_acc.update(trigger_pred.cpu(), trigger_label.cpu())
            cover_acc.update(cover_pred.cpu(), cover_img_label.cpu())
            dis_acc.update(torch.cat([fake_dis_pred, real_dis_true], dim=0).cpu(),
                           torch.cat([fake, valid], dim=0).cpu())
            precision.update(real_pred.cpu(), label.cpu())
            recall.update(real_pred.cpu(), label.cpu())
            f1.update(real_pred.cpu(), label.cpu())
            # 只是将loss的值保存暂存在字典中，方便后续用for循环进行update操作
            temp_losses_dict = {
                'loss_Dis': loss_Dis.item(),
                'loss_cat_Dnn': loss_cat_Dnn.item()
            }
            for tag, metric in temp_losses_dict.items():
                if tag == 'loss_cat_Dnn':
                    losses_dict[tag].update(metric, inputs.size(0))  # 注意分母, loss_cat_Dnn是合并输入的维度
                else:
                    losses_dict[tag].update(metric, trigger.size(0))  # 其他网络模块的损失函数，都是其输入(trigger)的维度
            # 更新PSNR
            batch_psnr = cal_psnr(cover_img.detach(), trigger.detach())
            img_quality_dict['psnr'].update(batch_psnr, trigger.size(0))

    logging.info('Loss Dis：{:.4f} Loss_cat_Dnn: {:.4f} Real acc: {:.4%} Trigger acc: {:.4%} Cover acc: {:.4%} '
                 'Precision: {:.4%} Recall: {:.4%} F1: {:.4%} PSNR: {:.2f}'.format(
        losses_dict['loss_Dis'].avg, losses_dict['loss_cat_Dnn'].avg,
        real_acc.compute(), trigger_acc.compute(), cover_acc.compute(),
        precision.compute(), recall.compute(), f1.compute(), batch_psnr))
    logging.info('test duration {:.2f} sec'.format(test_duration))

    plot_confusion_matrix(1, run_folder, 'test_real', real_preds, real_trues, test_loader)
    plot_confusion_matrix(1, run_folder, 'test_trigger', trigger_preds, trigger_trues, trigger_loader)
    plot_confusion_matrix(1, run_folder, 'test_dis', dis_preds, dis_trues, None)


# myepoch = 0
myepoch1 = {"wd": 0}


def wm_main():
    best_real_acc, best_trigger_acc, best_cover_acc, best_precision, best_recall, best_f1, \
    best_psnr, best_ssim = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    best_real_acc_epoch, best_trigger_acc_epoch, best_cover_acc_epoch, \
    best_precision_epoch, best_recall_epoch, best_f1_epoch, best_psnr_epoch, best_ssim_epoch = 0, 0, 0, 0, 0, 0, 0, 0
    # 如果有权重文件，就加载宿主模型及其优化器的权重
    ckp_path = os.path.join(cfg.watermark.ckp_path, ''.join(os.listdir(cfg.watermark.ckp_path)))
    if cfg.train.fine_tuning == True:
        checkpoint = torch.load(ckp_path)
        Dnnet.load_state_dict(checkpoint['model_state_dict'])
        optimizerN.load_state_dict(checkpoint['optimizer_state_dict'])
        logging.info("=> Load dnn checkpoint from '{}'".format(cfg.watermark.ckp_path))
    elif cfg.train.fine_tuning == False:
        pass

    global myepoch
    for epoch in range(cfg.train.start_epoch, cfg.train.num_epochs + 1):  # 注意epoch边界

        # myepoch = myepoch + 1
        myepoch1["wd"] += 1

        train_losses_dict, train_metrics_dict = train(epoch)

        val_losses_dict, val_metrics_dict, img_quality_dict, triggers, trigger_labels, trigger_ext_output = validation(
            epoch)
        plot_scalars(epoch, run_folder, train_losses_dict, train_metrics_dict, val_losses_dict, val_metrics_dict,
                     img_quality_dict)

        # 根据两个不同的标准挑选模型
        if val_metrics_dict['real_acc'] > best_real_acc:  # 强调尽量减少宿主模型任务的精度损失，保存这版模型(微调时可能不起作用)
            best_real_acc = val_metrics_dict['real_acc']
            best_real_acc_epoch = epoch
            save_checkpoint(epoch, run_folder, Hidnet, Disnet, Extnet, Dnnet, optimizerH, optimizerD, optimizerE,
                            optimizerN,
                            val_losses_dict, val_metrics_dict, img_quality_dict, 'real_acc_criteria', best_real_acc)

        if val_metrics_dict['trigger_acc'] > best_trigger_acc:  # 强调微调后宿主模型水印的安全性，保存这版模型并更新相应触发集图片
            best_trigger_acc = val_metrics_dict['trigger_acc']
            best_trigger_acc_epoch = epoch
            save_checkpoint(epoch, run_folder, Hidnet, Disnet, Extnet, Dnnet, optimizerH, optimizerD, optimizerE,
                            optimizerN,
                            val_losses_dict, val_metrics_dict, img_quality_dict, 'trigger_acc_criteria',
                            best_trigger_acc)

            save_separate_image(epoch, run_folder, triggers, trigger_labels, trigger_ext_output)

        # 以下指标只保存最优，不会根据这些指标挑选模型
        if img_quality_dict['psnr'].avg > best_psnr:
            best_psnr = img_quality_dict['psnr'].avg
            best_psnr_epoch = epoch
        if val_losses_dict['loss_ssim'].avg > best_ssim:
            best_ssim = val_losses_dict['loss_ssim'].avg
            best_ssim_epoch = epoch
        if val_metrics_dict['cover_acc'] > best_cover_acc:
            best_cover_acc = val_metrics_dict['cover_acc']
            best_cover_acc_epoch = epoch
        if val_metrics_dict['precision'] > best_precision:
            best_precision = val_metrics_dict['precision']
            best_precision_epoch = epoch
        if val_metrics_dict['recall'] > best_recall:
            best_recall = val_metrics_dict['recall']
            best_recall_epoch = epoch
        if val_metrics_dict['f1'] > best_f1:
            best_f1 = val_metrics_dict['f1']
            best_f1_epoch = epoch

    logging.info("################## Training and validation are finished! ##################")
    logging.info("In epoch {}: best psnr: {:.4%}".format(best_psnr_epoch, best_psnr))
    logging.info("In epoch {}: best ssim: {:.4%}".format(best_ssim_epoch, best_ssim))
    logging.info("In epoch {}: best real acc: {:.4%}".format(best_real_acc_epoch, best_real_acc))
    logging.info("In epoch {}: best trigger acc: {:.4%}".format(best_trigger_acc_epoch, best_trigger_acc))
    logging.info("In epoch {}: best cover acc: {:.4%}".format(best_cover_acc_epoch, best_cover_acc))
    logging.info("In epoch {}: best precision: {:.4%}".format(best_precision_epoch, best_precision))
    logging.info("In epoch {}: best recall: {:.4%}".format(best_recall_epoch, best_recall))
    logging.info("In epoch {}: best f1: {:.4%}".format(best_f1_epoch, best_f1))

    logging.info("################## Testing... ##################")
    test(run_folder, "runs/2022.01.20--22-54-52-55epoch")

# if __name__ == '__main__':
#     main()
