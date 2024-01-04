# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import torch
import torch.distributed as dist

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None

from argparse import ArgumentParser
from itertools import chain

import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision

from timm.models.layers import trunc_normal_

def load_checkpoint(config, model, optimizer, lr_scheduler, logger):
    logger.info(f"==============> Resuming form {config.MODEL.RESUME}....................")
    if config.MODEL.RESUME.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    logger.info(msg)
    max_accuracy = 0.0
    if not config.EVAL_MODE and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
        config.freeze()
        if 'amp' in checkpoint and config.AMP_OPT_LEVEL != "O0" and checkpoint['config'].AMP_OPT_LEVEL != "O0":
            amp.load_state_dict(checkpoint['amp'])
        logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")
        if 'max_accuracy' in checkpoint:
            max_accuracy = checkpoint['max_accuracy']

    del checkpoint
    torch.cuda.empty_cache()
    return max_accuracy

def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed


def load_pretrained(config, model, logger):
    logger.info(f"==============> Loading weight {config.MODEL.PRETRAINED} for fine-tuning......")

    if config.MODEL.TYPE == 'vit_mae':
        checkpoint = torch.load(config.MODEL.PRETRAINED, map_location='cpu')

        print("Load pre-trained checkpoint from: %s" % config.MODEL.PRETRAINED)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # print('######################, model')

        # print(model.state_dict().keys())

        # print('######################, ckpt')

        # print(checkpoint_model.keys())

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        if config.MODEL.TYPE == 'vit_base_patch16':

            # if args.global_pool:
            #     assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
            # else:
            #     assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

            # manually initialize fc layer
            trunc_normal_(model.head.weight, std=2e-5)

    else:
        # path = "/home/isalab103/kaoyu/CLIP/ViTAE-Transformer-Remote-Sensing/Scene_Recognition/pretrained/rsp-swin-t-ckpt.pth"
        # checkpoint = torch.load(path, map_location='cpu')
        checkpoint = torch.load(config.MODEL.PRETRAINED, map_location='cpu')

        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict_ema' in checkpoint:
            state_dict = checkpoint['state_dict_ema']  # imp_vitae
        else:
            state_dict = checkpoint

        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}

        if config.MODEL.TYPE != 'resnet':
            # delete relative_position_index since we always re-init it
            relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
            for k in relative_position_index_keys:
                del state_dict[k]

            # delete relative_coords_table since we always re-init it
            relative_position_index_keys = [k for k in state_dict.keys() if "relative_coords_table" in k]
            for k in relative_position_index_keys:
                del state_dict[k]

            # delete attn_mask since we always re-init it
            attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
            for k in attn_mask_keys:
                del state_dict[k]

            # bicubic interpolate relative_position_bias_table if not match
            relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
            for k in relative_position_bias_table_keys:
                relative_position_bias_table_pretrained = state_dict[k]
                relative_position_bias_table_current = model.state_dict()[k]
                L1, nH1 = relative_position_bias_table_pretrained.size()
                L2, nH2 = relative_position_bias_table_current.size()
                if nH1 != nH2:
                    logger.warning(f"Error in loading {k}, passing......")
                else:
                    if L1 != L2:
                        # bicubic interpolate relative_position_bias_table if not match
                        S1 = int(L1 ** 0.5)
                        S2 = int(L2 ** 0.5)
                        relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                            relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(S2, S2),
                            mode='bicubic')
                        state_dict[k] = relative_position_bias_table_pretrained_resized.view(nH2, L2).permute(1, 0)

            # bicubic interpolate absolute_pos_embed if not match
            absolute_pos_embed_keys = [k for k in state_dict.keys() if "absolute_pos_embed" in k]
            for k in absolute_pos_embed_keys:
                # dpe
                absolute_pos_embed_pretrained = state_dict[k]
                absolute_pos_embed_current = model.state_dict()[k]
                _, L1, C1 = absolute_pos_embed_pretrained.size()
                _, L2, C2 = absolute_pos_embed_current.size()
                if C1 != C1:
                    logger.warning(f"Error in loading {k}, passing......")
                else:
                    if L1 != L2:
                        S1 = int(L1 ** 0.5)
                        S2 = int(L2 ** 0.5)
                        absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.reshape(-1, S1, S1, C1)
                        absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.permute(0, 3, 1, 2)
                        absolute_pos_embed_pretrained_resized = torch.nn.functional.interpolate(
                            absolute_pos_embed_pretrained, size=(S2, S2), mode='bicubic')
                        absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.permute(0, 2, 3,
                                                                                                              1)
                        absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.flatten(1, 2)
                        state_dict[k] = absolute_pos_embed_pretrained_resized

            # check classifier, if not match, then re-init classifier to zero
            head_bias_pretrained = state_dict['head.bias']
            Nc1 = head_bias_pretrained.shape[0]
            Nc2 = model.head.bias.shape[0]
            if (Nc1 != Nc2):
                if Nc1 == 21841 and Nc2 == 1000:
                    logger.info("loading ImageNet-22K weight to ImageNet-1K ......")
                    map22kto1k_path = f'data/map22kto1k.txt'
                    with open(map22kto1k_path) as f:
                        map22kto1k = f.readlines()
                    map22kto1k = [int(id22k.strip()) for id22k in map22kto1k]
                    state_dict['head.weight'] = state_dict['head.weight'][map22kto1k, :]
                    state_dict['head.bias'] = state_dict['head.bias'][map22kto1k]
                else:
                    # 类别数不一致的话，且不是imagenet-22k->imagenet-1k，就不导入预训练参数，要重新trn linear层
                    torch.nn.init.constant_(model.head.bias, 0.)
                    torch.nn.init.constant_(model.head.weight, 0.)
                    del state_dict['head.weight']
                    del state_dict['head.bias']
                    logger.warning(f"Error in loading classifier head, re-init classifier head to 0")

        else:
            head_bias_pretrained = state_dict['fc.bias']
            Nc1 = head_bias_pretrained.shape[0]
            Nc2 = model.fc.bias.shape[0]
            if (Nc1 != Nc2):
                if Nc1 == 21841 and Nc2 == 1000:
                    logger.info("loading ImageNet-22K weight to ImageNet-1K ......")
                    map22kto1k_path = f'data/map22kto1k.txt'
                    with open(map22kto1k_path) as f:
                        map22kto1k = f.readlines()
                    map22kto1k = [int(id22k.strip()) for id22k in map22kto1k]
                    state_dict['head.weight'] = state_dict['head.weight'][map22kto1k, :]
                    state_dict['head.bias'] = state_dict['head.bias'][map22kto1k]
                else:
                    # 类别数不一致的话，且不是imagenet-22k->imagenet-1k，就不导入预训练参数，要重新trn linear层
                    torch.nn.init.constant_(model.fc.bias, 0.)
                    torch.nn.init.constant_(model.fc.weight, 0.)
                    del state_dict['fc.weight']
                    del state_dict['fc.bias']
                    logger.warning(f"Error in loading classifier head, re-init classifier head to 0")

        msg = model.load_state_dict(state_dict, strict=False)
        logger.warning(msg)

    logger.info(f"=> loaded successfully '{config.MODEL.PRETRAINED}'")

    del checkpoint
    torch.cuda.empty_cache()


def save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, logger):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'epoch': epoch,
                  'config': config}
    if config.AMP_OPT_LEVEL != "O0":
        save_state['amp'] = amp.state_dict()

    save_path = os.path.join(config.OUTPUT, 'ckpt.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt
