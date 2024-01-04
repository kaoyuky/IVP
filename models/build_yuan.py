# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from .swin_transformer import *
#from .swin_mlp import SwinMLP
from .ViTAE_Window_NoShift.models import ViTAE_Window_NoShift_12_basic_stages4_14
from .resnet import resnet50
from .vit_win_rvsa import ViT_Win_RVSA
from .vit_mae import *

def build_model(config):
    model_type = config.MODEL.TYPE
    if model_type == 'swin':
        model = PromptedSwinTransformer(img_size=config.DATA.IMG_SIZE,
                                patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                in_chans=config.MODEL.SWIN.IN_CHANS,
                                num_classes=config.MODEL.NUM_CLASSES,
                                embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                depths=config.MODEL.SWIN.DEPTHS,
                                num_heads=config.MODEL.SWIN.NUM_HEADS,
                                window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                qk_scale=config.MODEL.SWIN.QK_SCALE,
                                drop_rate=config.MODEL.DROP_RATE,
                                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                ape=config.MODEL.SWIN.APE,
                                patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                use_checkpoint=config.TRAIN.USE_CHECKPOINT)
    elif model_type == 'resnet':

        model = resnet50(num_classes=config.MODEL.NUM_CLASSES)

        print('Using ResNet50!')

    elif model_type == 'vitae_win':

        model = ViTAE_Window_NoShift_12_basic_stages4_14(img_size=config.DATA.IMG_SIZE, num_classes=config.MODEL.NUM_CLASSES, window_size=7)

        print('Using VitAE_v2!')

    elif model_type == 'vit_mae':

        model = PromptedViT_Win_RVSA(config.MODEL.PROMPT, img_size=config.DATA.IMG_SIZE, num_classes=config.MODEL.NUM_CLASSES, drop_path_rate=0.1,
                             use_abs_pos_emb=True, interval=1)

        print('Using Vit_MAE!')
        
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
