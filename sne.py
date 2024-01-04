# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import time
import random
import argparse
import datetime
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter

from config import get_config
from models import build_model
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, load_pretrained, save_checkpoint, get_grad_norm, auto_resume_helper, reduce_tensor

from sklearn.manifold import TSNE
# from time import time
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None


def parse_option():
    parser = argparse.ArgumentParser('Remote sensing training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=False, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--data-path', default=None, type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O0', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    # batch
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    # epoch
    parser.add_argument('--epochs', type=int, help="epochs")
    # distributed training
    parser.add_argument("--local_rank", default=0, type=int, help='local rank for DistributedDataParallel')
    # dataset
    parser.add_argument('--dataset', default=None, type=str, choices=['millionAID','ucm','aid','nwpuresisc'], help='type of dataset')
    # ratio
    parser.add_argument('--ratio', default=None, type=int, help='trn tes ratio')
    # model
    parser.add_argument('--model', default=None, type=str, choices=['resnet','vit','swin','vitae_win','vit_mae'], help='type of model')
    # input size
    parser.add_argument("--img_size", default=None, type=int, help='size of input')
    # exp_num
    parser.add_argument("--exp_num", default=0, type=int, help='number of experiment times')
    # tag
    parser.add_argument("--split", default=None, type=int, help='id of split')
    # lr
    parser.add_argument("--lr", default=None, type=float, help='learning rate')
    # wd
    parser.add_argument("--weight_decay", default=None, type=float, help='learning rate')
    # gpu_num
    parser.add_argument("--gpu_num", default=None, type=int, help='id of split')
    # optimizer
    parser.add_argument('--optimizer', default='adamw', type=str, choices=['adamw','sgd'], help='type of optimizer')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def set_plt(start_time, end_time, title):
    plt.title(f'{title} time consume:{end_time - start_time:.3f} s')
    # plt.legend(title='')
    plt.ylabel('')
    plt.xlabel('')
    plt.xticks([])
    plt.yticks([])

def t_sne(data, label, title):
    # t-sne处理
    print('starting T-SNE process')
    start_time = time.time()
    data = TSNE(n_components=2, learning_rate='auto', init='pca').fit_transform(data)
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    # data = (data - x_min) / (x_max - x_min)
    df = pd.DataFrame(data, columns=['x', 'y'])  # 转换成df表
    df.insert(loc=1, column='label', value=label)
    end_time = time.time()
    print('Finished')

    # 绘图
    ax_total_message_ratio = sns.scatterplot(x='x', y='y', hue='label', s=1, palette="Set2", data=df)
    ax_total_message_ratio.get_legend().remove()  # .legend(loc="lower right")
    # sns.scatterplot(x='x', y='y', s=6, palette="Set2", data=df)
    set_plt(start_time, end_time, title)
    plt.savefig('1.jpg', dpi=400)
    plt.show()


def main(args, config):
    config.defrost()  # 释放cfg

    if config.MODEL.TYPE == 'resnet':
        config.MODEL.NAME = 'resnet_50_224'
    elif config.MODEL.TYPE == 'swin':
        config.MODEL.NAME = 'swin_tiny_patch4_window7_224'
    elif config.MODEL.TYPE == 'vitae_win':
        config.MODEL.NAME = 'ViTAE_Window_NoShift_12_basic_stages4_14_224'
    elif config.MODEL.TYPE == 'vit_mae':
        config.MODEL.NAME = 'ViT_MAE'

    if config.DATA.DATASET == 'millionAID':
        config.DATA.DATA_PATH = '../Dataset/millionaid/'
        config.MODEL.NUM_CLASSES = 51
    elif config.DATA.DATASET == 'ucm':
        config.DATA.DATA_PATH = '../Dataset/ucm/'
        config.MODEL.NUM_CLASSES = 21
    elif config.DATA.DATASET == 'aid':
        config.DATA.DATA_PATH = '../Dataset/aid/'
        config.MODEL.NUM_CLASSES = 30
    elif config.DATA.DATASET == 'nwpuresisc':
        config.DATA.DATA_PATH = '../Dataset/nwpu_resisc45/'
        config.MODEL.NUM_CLASSES = 45

    config.freeze()

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    # model.head = torch.nn.Sequential(torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6), model.head)
    model.cuda()
    logger.info(str(model))

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False,
                                                      find_unused_parameters=True)
    model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model_without_ddp, 'flops'):
        flops = model_without_ddp.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")


    for i in range(args.exp_num):

        seed = i + 2022
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        cudnn.benchmark = True

        dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config, args.ratio,
                                                                                                logger, args.split)

        acc1, acc5, loss = validate(config, data_loader_val, model)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        if config.EVAL_MODE:
            return



@torch.no_grad()
def validate(config, data_loader, model):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()

    x_tsne = []
    y_tsne = []

    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output, x_sem = model(images)

        targets = target.clone()
        # targets[targets == 0] = 1
        # targets[targets == -1] = 0
        x_tsne.append(x_sem)
        y_tsne.append(targets)


        # measure accuracy and record loss
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        acc1 = reduce_tensor(acc1)
        acc5 = reduce_tensor(acc5)
        loss = reduce_tensor(loss)

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')

    feature_map_list = torch.cat(x_tsne, dim=0)
    y_tsne = torch.cat(y_tsne, dim=0)
    # print(feature_map_list.shape)
    t_sne(data=np.array(feature_map_list.cpu()), label=np.array(y_tsne.cpu()), title=f'resnet feature map\n')
    

    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg


if __name__ == '__main__':
    args, config = parse_option()

    if config.AMP_OPT_LEVEL != "O0":
        assert amp is not None, "amp not installed!"

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    # seed = config.SEED + dist.get_rank()

    opt_lower = config.TRAIN.OPTIMIZER.NAME.lower()

    if opt_lower == 'adamw':
        # linear scale the learning rate according to total batch size, may not be optimal
        linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
        linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
        linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
        # gradient accumulation also need to scale the learning rate
        if config.TRAIN.ACCUMULATION_STEPS > 1:
            linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
            linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
            linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS

        config.defrost()
        config.TRAIN.BASE_LR = linear_scaled_lr
        config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
        config.TRAIN.MIN_LR = linear_scaled_min_lr
        config.freeze()

    elif opt_lower == 'sgd':
        config.defrost()
        config.TRAIN.LR_SCHEDULER.NAME = 'step'
        config.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = config.TRAIN.EPOCHS // 3 - config.TRAIN.EPOCHS // 10
        config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())

    main(args, config)
