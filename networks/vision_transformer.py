# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from .swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys
from .Swin_Unet_25D import SwinTransformer25D
logger = logging.getLogger(__name__)

class SwinUnet(nn.Module):
    def __init__(self, model_25D, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(SwinUnet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.config = config
        self.model_25D = model_25D
        if self.model_25D:
            print('Using swin-unet 2.5D backbone')
            self.swin_unet = SwinTransformer25D(img_size=config.DATA.IMG_SIZE,
                                                patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                                in_chans=config.MODEL.SWIN.IN_CHANS,
                                                num_classes=self.num_classes,
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
        else:
            print('Using swin-unet 2D backbone')
            self.swin_unet = SwinTransformerSys(img_size=config.DATA.IMG_SIZE,
                                    patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                    in_chans=config.MODEL.SWIN.IN_CHANS,
                                    num_classes=self.num_classes,
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

    def forward(self, x, train=True, idx_slice=-1, stride=-1, length=-1):
        if self.model_25D:
            logits = self.swin_unet(x, train, idx_slice, stride, length)
        else:
            if x.size()[1] == 1:
                x = x.repeat(1,3,1,1)
            logits = self.swin_unet(x)
        return logits

    def load_from(self, config):
        pretrained_path = config.MODEL.PRETRAIN_CKPT
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model"  not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]:v for k,v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                msg = self.swin_unet.load_state_dict(pretrained_dict,strict=False)
                # print(msg)
                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")

            model_dict = self.swin_unet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                # if k=='norm.weight' or k=='norm.bias':
                #     current_k1 = k.replace('norm', 'norm1')
                #     current_k2 = k.replace('norm', 'norm2')
                #     current_k3 = k.replace('norm', 'norm3')
                #     full_dict.update({current_k1: v})
                #     full_dict.update({current_k2: v})
                #     full_dict.update({current_k3: v})
                # if "patch_embed." in k:
                #     current_p_LR = k.replace('patch_embed', 'patch_embed_LR')
                #     current_p_M = k.replace('patch_embed', 'patch_embed_M')
                #     full_dict.update({current_p_LR: v})
                #     full_dict.update({current_p_M: v})
                if "layers." in k:
                    current_layer_num = 3-int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k: v})
                    # current_k_LR = k.replace('layers', 'layers_LR')
                    # current_k_M = k.replace('layers', 'layers_M')
                    # full_dict.update({current_k_LR: v})
                    # full_dict.update({current_k_M: v})
            for k in list(full_dict.keys()):
                # print('pre-train key:', k)
                if k in model_dict.keys():
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
                        del full_dict[k]
                else:
                    pass
                    # print('Unknown key:', k)
            msg = self.swin_unet.load_state_dict(full_dict, strict=False)
            # print(msg)
        else:
            print("none pretrain")
