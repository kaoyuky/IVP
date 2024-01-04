#!/usr/bin/env python3
"""
swin transformer with prompt
"""
import math
from simplejson import OrderedDict
import torch
import torch.nn as nn
import torchvision as tv

from functools import reduce
from operator import mul
from torch.nn import Conv2d, Dropout

from timm.models.layers import to_2tuple
import torch.nn.functional as F

from .NormalCell import NormalCell
from .ReductionCell import ReductionCell

from .base_model import *
from .swin import (SwinTransformer, SwinTransformerBlock,
    window_partition, window_reverse, WindowAttention)
from .token_transformer import *
from .token_performer import *
from ..Logging import *
logger = get_logger("visual_prompt")


class PromptGeneratorWithNaive(nn.Module):
    def __init__(self, input: int, downsize: int, prompt: int):
        super().__init__()
        self.proj_down = nn.Linear(input, input // downsize)
        self.intermediate_act_fn = nn.ReLU()
        self.proj_up = nn.Linear(input // downsize, prompt * input)
        # self.norm1 = nn.LayerNorm(input)
        self.norm2 = nn.BatchNorm1d(input//downsize)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.num_prompt_tokens = prompt
        self.dropout = nn.Dropout(0.2)

    def forward(self, hidden_states=None, attention_mask=None):
        sequence_output = hidden_states

        # sequence_output = self.norm1(hidden_states)
        sequence_output = (self.avgpool(sequence_output.transpose(1, 2))).squeeze()
        # print(sequence_output.shape)
        hidden_states = self.proj_down(sequence_output)
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.proj_up(hidden_states)
        hidden_states = hidden_states.view(sequence_output.shape[0], self.num_prompt_tokens, -1)


        return hidden_states


class PromptGeneratorWithPooling(nn.Module):
    def __init__(self, input: int, downsize: int, prompt: int):
        super().__init__()

        self.proj_down = nn.Linear(input, input // downsize)
        self.intermediate_act_fn = nn.ReLU()
        self.proj_up = nn.Linear(input // downsize, input)
        # self.norm1 = nn.LayerNorm(input)
        self.norm2 = nn.BatchNorm1d(input // downsize)
        # self.norm2 = nn.BatchNorm1d(prompt)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.num_prompt_tokens = prompt
        self.dropout = nn.Dropout(0.2)
        self.generator_type = 'MPPG'

        if self.generator_type == 'MPPG':
            self.adaptive_pooling = nn.AdaptiveMaxPool1d(self.num_prompt_tokens)
        elif self.generator_type == 'APPG':
            self.adaptive_pooling = nn.AdaptiveAvgPool1d(self.num_prompt_tokens)


    def forward(self, hidden_states=None, attention_mask=None):
        hidden_states = self.proj_down(hidden_states)

        batch_prompts = []
        for i in range(hidden_states.size(0)):
            hidden_state = hidden_states[i].unsqueeze(0)
            hidden_state = hidden_state.transpose(1, 2) # B x D x L
            hidden_state = (self.adaptive_pooling(hidden_state)).transpose(1, 2) # B x num_prompt_tokens x D
            batch_prompts.append(hidden_state)

        hidden_states = torch.cat(batch_prompts, dim=0)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.norm2(hidden_states.transpose(1, 2)).transpose(1, 2)
        hidden_states = self.proj_up(hidden_states)

        return hidden_states


class PromptedViTAE_Window_NoShift_basic(ViTAE_Window_NoShift_basic):
    def __init__(
            self, cfg, img_size=224, in_chans=3, stages=4, embed_dims=64, token_dims=64, downsample_ratios=[4, 2, 2, 2], kernel_size=[7, 3, 3, 3],
            RC_heads=[1, 1, 1, 1], NC_heads=4, dilations=[[1, 2, 3, 4], [1, 2, 3], [1, 2], [1, 2]],
            RC_op='cat', RC_tokens_type=['performer', 'transformer', 'transformer', 'transformer'], NC_tokens_type='transformer',
            RC_group=[1, 1, 1, 1], NC_group=[1, 32, 64, 64], NC_depth=[2, 2, 6, 2], mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0.,
            attn_drop_rate=0., drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=1000,
            gamma=False, init_values=1e-4, SE=False, window_size=7, relative_pos=False):

        self.prompt_LOCATION = cfg.LOCATION  # fansghi1
        self.prompt_NUM_TOKENS = cfg.NUM_TOKENS  # + self.prompt_NUM_0
        self.prompt_DROPOUT = cfg.DROPOUT
        self.prompt_PROJECT = cfg.PROJECT
        self.prompt_DEEP = cfg.DEEP
        self.prompt_INITIATION = cfg.INITIATION
        # self.depths = depths#np.sum(self.NC_depth)

        if self.prompt_LOCATION == "pad":
            img_size += 2 * self.prompt_NUM_TOKENS
        super(PromptedViTAE_Window_NoShift_basic, self).__init__(
            img_size, in_chans, stages, embed_dims, token_dims, downsample_ratios, kernel_size,
            RC_heads, NC_heads, dilations,
            RC_op, RC_tokens_type, NC_tokens_type,
            RC_group, NC_group, NC_depth, mlp_ratio, qkv_bias, qk_scale, drop_rate,
            attn_drop_rate, drop_path_rate, norm_layer, num_classes,
            gamma, init_values, SE, window_size, relative_pos
        )

        # img_size = to_2tuple(img_size)
        patch_size = to_2tuple(4)
        if self.prompt_LOCATION == "add":
            num_tokens = self.embeddings.position_embeddings.shape[1]
        elif self.prompt_LOCATION == "add-1":
            num_tokens = 1
        else:
            num_tokens = self.prompt_NUM_TOKENS

        self.prompt_dropout = Dropout(self.prompt_DROPOUT)
        # if project the prompt embeddings
        if self.prompt_PROJECT > -1:
            # only for prepend / add
            prompt_dim = self.prompt_PROJECT
            self.prompt_proj = nn.Linear(
                prompt_dim, embed_dims)
            nn.init.kaiming_normal_(
                self.prompt_proj.weight, a=0, mode='fan_out')
        else:
            self.prompt_proj = nn.Identity()

        depth = np.sum(self.NC_depth)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        Layers = []
        for i in range(stages):
            startDpr = 0 if i == 0 else self.NC_depth[i - 1]
            Layers.append(
                BasicLayer(img_size, in_chans, self.embed_dims[i], self.tokens_dims[i], self.downsample_ratios[i],
                           self.kernel_size[i], self.RC_heads[i], self.NC_heads[i], self.dilaions[i], self.RC_op[i],
                           self.RC_tokens_type[i], self.NC_tokens_type[i], self.RC_group[i], self.NC_group[i],
                           self.NC_depth[i], dpr[startDpr:self.NC_depth[i] + startDpr],
                           mlp_ratio=self.mlp_ratio[i], qkv_bias=self.qkv_bias[i], qk_scale=self.qk_scale[i],
                           drop=self.drop[i], attn_drop=self.attn_drop[i],
                           norm_layer=self.norm_layer[i], gamma=gamma, init_values=init_values, SE=SE,
                           window_size=window_size, relative_pos=relative_pos,
                           RC=None, NC=PromptedNormalCell,
                           num_prompts=num_tokens, prompt_location=self.prompt_LOCATION, deep_prompt=self.prompt_DEEP)
            )
            img_size = img_size // self.downsample_ratios[i]
            in_chans = self.tokens_dims[i]
        self.layers = nn.ModuleList(Layers)

        if self.prompt_INITIATION == "random":
            if self.prompt_LOCATION == "prepend":
                val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + self.tokens_dims[0]))  # noqa
                if self.prompt_DEEP:
                    # NOTE: only for 4 layers, need to be more flexible
                    self.deep_prompt_embeddings_0 = nn.Parameter(
                        torch.zeros(
                            NC_depth[0], num_tokens, self.tokens_dims[0]
                        ))
                    nn.init.uniform_(
                        self.deep_prompt_embeddings_0.data, -val, val)
                    self.deep_prompt_embeddings_1 = nn.Parameter(
                        torch.zeros(
                            NC_depth[1], num_tokens, self.tokens_dims[1]
                        ))
                    nn.init.uniform_(
                        self.deep_prompt_embeddings_1.data, -val, val)
                    self.deep_prompt_embeddings_2 = nn.Parameter(
                        torch.zeros(
                            NC_depth[2], num_tokens, self.tokens_dims[2]
                        ))
                    nn.init.uniform_(
                        self.deep_prompt_embeddings_2.data, -val, val)
                    self.deep_prompt_embeddings_3 = nn.Parameter(
                        torch.zeros(
                            NC_depth[3], num_tokens, self.tokens_dims[3]
                        ))
                    nn.init.uniform_(
                        self.deep_prompt_embeddings_3.data, -val, val)
                else:
                    self.deep_prompt_embeddings_0 = nn.Parameter(
                        torch.zeros(
                            1, num_tokens, self.tokens_dims[0]
                        ))
                    nn.init.uniform_(
                        self.deep_prompt_embeddings_0.data, -val, val)

            else:
                raise ValueError("Other prompt locations are not supported")
        else:

            if self.prompt_DEEP:
                self.deep_prompt_embeddings_0 = nn.ModuleList()
                for i_layer in range(NC_depth[0]):
                    layer = PromptGeneratorWithPooling(self.tokens_dims[0], cfg.project, num_tokens)
                    self.deep_prompt_embeddings_0.append(layer)

                self.deep_prompt_embeddings_1 = nn.ModuleList()
                for i_layer in range(NC_depth[1]):
                    layer = PromptGeneratorWithPooling(self.tokens_dims[1], cfg.project, num_tokens)
                    self.deep_prompt_embeddings_1.append(layer)

                self.deep_prompt_embeddings_2 = nn.ModuleList()
                for i_layer in range(NC_depth[2]):
                    layer = PromptGeneratorWithPooling(self.tokens_dims[2], cfg.project, num_tokens)
                    self.deep_prompt_embeddings_2.append(layer)

                self.deep_prompt_embeddings_3 = nn.ModuleList()
                for i_layer in range(NC_depth[3]):
                    layer = PromptGeneratorWithPooling(self.tokens_dims[3], cfg.project, num_tokens)
                    self.deep_prompt_embeddings_3.append(layer)
            else:
                self.deep_prompt_embeddings_0 = PromptGeneratorWithNaive(self.tokens_dims[0], cfg.project,
                                                                          num_tokens)



    def incorporate_prompt(self, x):
        B = x.shape[0]
        prompt_emb_lr = self.prompt_norm(
            self.prompt_embeddings0_lr).expand(B, -1, -1, -1)
        prompt_emb_tb = self.prompt_norm(
            self.prompt_embeddings0_tb).expand(B, -1, -1, -1)

        x = torch.cat((
            prompt_emb_lr[:, :, :, :5],
            x, prompt_emb_lr[:, :, :, 5:]
        ), dim=-1)
        x = torch.cat((
            prompt_emb_tb[:, :, :5, :],
            x, prompt_emb_tb[:, :, 5:, :]
        ), dim=-2)
        # x = self.prompt_layers(x)
        return x

    def forward_features(self, x):
        if self.prompt_LOCATION == "prepend" and self.prompt_DEEP:
            # x = self.incorporate_prompt(x)
            for layer, NC_prompt_embd in zip(
                    self.layers, [
                        self.deep_prompt_embeddings_0,
                        self.deep_prompt_embeddings_1,
                        self.deep_prompt_embeddings_2,
                        self.deep_prompt_embeddings_3
                    ],
            ):
                x = layer(x, NC_prompt_embd, self.prompt_dropout, self.prompt_INITIATION)
        else:
            for layer in self.layers:
                x = layer(x, self.deep_prompt_embeddings_0, self.prompt_dropout, self.prompt_INITIATION)

        return torch.mean(x, 1)



    def train(self, mode=True, tag='default'):
        self.training = mode
        # if tag == 'default':
        #     for module in self.modules():
        #         if module.__class__.__name__ != 'PromptedViTAE_Window_NoShift_basic':
        #             module.train(mode)
        if tag == 'default':
            if mode:
                for module in self.modules():
                    if module.__class__.__name__ != 'PromptedViTAE_Window_NoShift_basic':
                        module.train(False)
                    self.prompt_proj.train()
                    self.prompt_dropout.train()
            else:
                for module in self.modules():
                    if module.__class__.__name__ != 'PromptedViTAE_Window_NoShift_basic':
                        module.train(mode)
        elif tag == 'linear':
            for module in self.modules():
                if module.__class__.__name__ != 'PromptedViTAE_Window_NoShift_basic':
                    module.eval()
                    for param in module.parameters():
                        param.requires_grad = False
        elif tag == 'linearLNBN':
            for module in self.modules():
                if module.__class__.__name__ != 'PromptedViTAE_Window_NoShift_basic':
                    if isinstance(module, nn.LayerNorm) or isinstance(module, nn.BatchNorm2d):
                        module.train(mode)
                        for param in module.parameters():
                            param.requires_grad = True
                    else:
                        module.eval()
                        for param in module.parameters():
                            param.requires_grad = False
        self.head.train(mode)
        for param in self.head.parameters():
            param.requires_grad = True
        return self



class PromptedReductionCell(ReductionCell):
    def __init__(self, num_prompts, prompt_location, deep_prompt, img_size=224, in_chans=3, embed_dims=64, token_dims=64, downsample_ratios=4, kernel_size=7,
                 num_heads=1, dilations=[1,2,3,4], share_weights=False, op='cat', tokens_type='performer', group=1,
                 relative_pos=False, drop=0., attn_drop=0., drop_path=0., mlp_ratio=1.0, gamma=False, init_values=1e-4, SE=False, window_size=7
    ):
        super(PromptedReductionCell, self).__init__(
            img_size, in_chans, embed_dims, token_dims, downsample_ratios, kernel_size,
            num_heads, dilations, share_weights, op, tokens_type, group,
            relative_pos, drop, attn_drop, drop_path, mlp_ratio, gamma, init_values,
            SE, window_size)

        self.num_prompts = num_prompts
        self.prompt_location = prompt_location

        # self.pr

        in_chans = self.PRM.out_chans
        if self.prompt_location == "prepend":
            if tokens_type == 'performer':
                # assert num_heads == 1
                self.attn = Token_performer(dim=in_chans, in_dim=token_dims, head_cnt=num_heads, kernel_ratio=0.5,
                                            gamma=gamma, init_values=init_values)
            elif tokens_type == 'performer_less':
                self.attn = None
                self.PCM = None
            elif tokens_type == 'transformer':
                self.attn = Token_transformer(dim=in_chans, in_dim=token_dims, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                              drop=drop,
                                              attn_drop=attn_drop, drop_path=drop_path, gamma=gamma,
                                              init_values=init_values)
            elif tokens_type == 'swin':
                self.attn = SwinTransformerBlock(
                    in_dim=in_chans, out_dim=token_dims,
                    input_resolution=(self.img_size // self.downsample_ratios, self.img_size // self.downsample_ratios),
                    num_heads=num_heads, mlp_ratio=mlp_ratio, drop=drop,
                    attn_drop=attn_drop, drop_path=drop_path, window_size=window_size, shift_size=0,
                    relative_pos=relative_pos)


    def forward(self, x, prompt_emb_lr=None):
        # B, L, C = x.shape
        # if self.prompt_location == "prepend":
        #     # change input size
        #     prompt_emb = x[:, :self.num_prompts, :]
        #     x = x[:, self.num_prompts:, :]
        #     L = L - self.num_prompts

        # prompt_embd = prompt_embd
        # NUM = True
        if len(x.shape) < 4:

            if self.prompt_location == "prepend":
                # change input size
                # prompt_emb = x[:, :self.num_prompts, :]
                x = x[:, self.num_prompts:, :]

            # B,N,C -> B,H,W,C
            B, N, C = x.shape
            n = int(np.sqrt(N))
            x = x.view(B, n, n, C).contiguous()
            x = x.permute(0, 3, 1, 2)
            # NUM = False

        # x = torch.cat((
        #     prompt_emb_lr[:, :, :, :5],
        #     x, prompt_emb_lr[:, :, :, 5:]
        # ), dim=-1)
        # x = torch.cat((
        #     prompt_emb_tb[:, :, :5, :],
        #     x, prompt_emb_tb[:, :, 5:, :]
        # ), dim=-2)
        B, C1 = x.shape[0], prompt_emb_lr.shape[1]
        prompt_embd = prompt_emb_lr.unsqueeze(dim=1).expand(B, C1, -1, -1).permute(0, 3, 1, 2)

        if self.pool is not None:
            x = self.pool(x)
        shortcut = x
        PRM_x, _ = self.PRM(x)
        # B = x.shape[0]
        # prompt_embd = prompt_emb_lr.unsqueeze(dim=1).expand(B, self.num_prompts, -1, -1).permute(0, 3, 1, 2)
        # prompt_embd = prompt_emb_lr.reshape(1, self.num_prompts)
        prompt_emb, _ = self.PRM(prompt_embd)
        # prompt_emb = prompt_emb.mean(dim=1)
        # x1, x2, x3 = prompt_emb.shape
        # prompt_emb = prompt_emb.unsqueeze(dim=0)
        # prompt_emb = F.interpolate(prompt_emb, size=(self.num_prompts, x3), mode='bilinear', align_corners=True)
        # prompt_emb = prompt_emb.squeeze()
        num_prompts = prompt_emb.shape[1]
        if self.tokens_type == 'swin':
            pass
            B, N, C = PRM_x.shape
            H, W = self.img_size // self.downsample_ratios, self.img_size // self.downsample_ratios
            b, _, c = PRM_x.shape
            assert N == H*W
            x = self.attn.norm1(PRM_x)
            x = x.view(B, H, W, C)
            x_windows = window_partition(x, self.window_size)
            x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

            num_windows = int(x_windows.shape[0] / B)
            # print(prompt_emb.shape)
            # print(x_windows.shape)
            if self.prompt_location == "prepend":
                # expand prompts_embs
                # B, num_prompts, C --> nW*B, num_prompts, C
                prompt_emb = prompt_emb.unsqueeze(0)
                prompt_emb = prompt_emb.expand(num_windows, -1, -1, -1)
                prompt_emb = prompt_emb.reshape((-1, num_prompts, C))
                # print(prompt_emb.shape)
                # print(x_windows.shape)
                x_windows = torch.cat((prompt_emb, x_windows), dim=1)

            # print(x_windows.shape)
            attn_windows = self.attn.attn(x_windows, mask=self.attn.attn_mask)  # nW*B, window_size*window_size, C

            # seperate prompt embs --> nW*B, num_prompts, C
            if self.prompt_location == "prepend":
                # change input size
                prompt_emb = attn_windows[:, :num_prompts, :]
                attn_windows = attn_windows[:, num_prompts:, :]
                # change prompt_embs's shape:
                # nW*B, num_prompts, C - B, num_prompts, C
                # print(prompt_emb.shape)
                B1, N1, C1 = prompt_emb.shape
                prompt_emb = prompt_emb.view(-1, B, num_prompts, C1)
                prompt_emb = prompt_emb.mean(0)
                # print(prompt_emb.shape)

            attn_windows = attn_windows.view(-1, self.window_size, self.window_size, self.token_dims)
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
            x = shifted_x
            x1 = x.view(B, H * W, self.token_dims)

            x = torch.cat((prompt_emb, x1), dim=1)

            convX_x = self.PCM(shortcut)
            # print(shortcut.shape)
            # print(prompt_emb_lr.unsqueeze(dim=1).permute(0, 3, 1, 2).shape)
            # B,C,H,W->B,H,W,C-> B,H*W,C->B,L,C
            convX_x = convX_x.permute(0, 2, 3, 1).view(*x1.shape).contiguous()
            convX_p = self.PCM(prompt_embd)
            # y = convX_p.shape[2]
            # convX_p = F.interpolate(convX_p, size=(y, self.num_prompts), mode='bilinear', align_corners=True)
            # print(prompt_embd.shape)
            # print(convX_p.shape)
            convX_p = convX_p.permute(0, 2, 3, 1).view(*prompt_emb.shape).contiguous()
            convX = torch.cat((convX_p, convX_x), dim=1)
            x = x + self.attn.drop_path(convX * self.gamma2)

            # # add the prompt back:
            # if self.prompt_location == "prepend":
            #     # print(prompt_emb.shape)
            #     # print(x.shape)
            #     x = torch.cat((prompt_emb, x), dim=1)

            # x = shortcut + self.attn.drop_path(x)
            # x = x + self.attn.drop_path(self.attn.mlp(self.attn.norm2(x)))
            x = x + self.attn.drop_path(self.gamma3 * self.attn.mlp(self.attn.norm2(x)))
        else:
            if self.attn is None:
                return PRM_x
            convX_x = self.PCM(shortcut)

            PRM_x = torch.cat((prompt_emb, PRM_x), dim=1)

            x = self.attn.attn(self.attn.norm1(PRM_x))
            x1 = x[:, num_prompts:, :]
            convX_x = convX_x.permute(0, 2, 3, 1).view(*x1.shape).contiguous()
            # convX_p = self.PCM(prompt_emb_lr.unsqueeze(dim=1).permute(0, 3, 1, 2))
            convX_p = self.PCM(prompt_embd)
            # y = convX_p.shape[2]
            # convX_p = F.interpolate(convX_p, size=(y, self.num_prompts), mode='bilinear', align_corners=True)
            # print(convX_x.shape)
            # print(convX_p.shape)
            convX_p = convX_p.permute(0, 2, 3, 1).view(*prompt_emb.shape).contiguous()
            convX = torch.cat((convX_p, convX_x), dim=1)
            x = x + self.attn.drop_path(convX * self.gamma2)
            x = x + self.attn.drop_path(self.gamma3 * self.attn.mlp(self.attn.norm2(x)))
        x = self.SE(x)

        # shortcut = x
        # # print(shortcut.shape)
        # PRM_x, _ = self.PRM(x)
        # if self.tokens_type == 'swin':
        #     pass
        #     B, N, C = PRM_x.shape
        #     H, W = self.img_size // self.downsample_ratios, self.img_size // self.downsample_ratios
        #     H1, W1 = (self.img_size+10) // self.downsample_ratios, (self.img_size+10) // self.downsample_ratios
        #     b, _, c = PRM_x.shape
        #     # assert N == H * W
        #     x = self.attn.norm1(PRM_x)
        #     if NUM is True:
        #         x = x.view(B, H1, W1, C)
        #         x = F.interpolate(x.permute(0, 3, 1, 2), size=(H, W), mode='bilinear', align_corners=True).permute(0, 2,
        #                                                                                                            3, 1)
        #     else:
        #         x = x.view(B, H, W, C)
        #     # print(x.shape)
        #
        #     # x = F.interpolate(x.permute(0, 3, 1, 2), size=(H, W), mode='bilinear', align_corners=True).permute(0, 2, 3, 1)
        #
        #     x_windows = window_partition(x, self.window_size)
        #     x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        #     attn_windows = self.attn.attn(x_windows, mask=self.attn.attn_mask)  # nW*B, window_size*window_size, C
        #     attn_windows = attn_windows.view(-1, self.window_size, self.window_size, self.token_dims)
        #     shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
        #     x = shifted_x
        #     # print(x.shape)
        #     x = x.view(B, H * W, self.token_dims)
        #
        #     convX = self.PCM(shortcut)
        #     # B,C,H,W->B,H,W,C-> B,H*W,C->B,L,C
        #     convX = convX.permute(0, 2, 3, 1)
        #     if NUM is True:
        #         convX = F.interpolate(convX.permute(0, 3, 1, 2), size=(H, W), mode='bilinear',
        #                               align_corners=True).permute(0, 2, 3, 1)
        #     # convX = F.interpolate(convX.permute(0, 3, 1, 2), size=(H, W), mode='bilinear', align_corners=True).permute(0, 2, 3, 1)
        #     convX = convX.view(*x.shape).contiguous()
        #     x = x + self.attn.drop_path(convX * self.gamma2)
        #     # x = shortcut + self.attn.drop_path(x)
        #     # x = x + self.attn.drop_path(self.attn.mlp(self.attn.norm2(x)))
        #     x = x + self.attn.drop_path(self.gamma3 * self.attn.mlp(self.attn.norm2(x)))
        # else:
        #     if self.attn is None:
        #         return PRM_x
        #
        #     B, N, C = PRM_x.shape
        #     H, W = self.img_size // self.downsample_ratios, self.img_size // self.downsample_ratios
        #     H1, W1 = (self.img_size + 10) // self.downsample_ratios, (self.img_size + 10) // self.downsample_ratios
        #     # print(x.shape)
        #
        #     if NUM is True:
        #         PRM_x = PRM_x.view(B, H1, W1, C)
        #         PRM_x = F.interpolate(PRM_x.permute(0, 3, 1, 2), size=(H, W), mode='bilinear',
        #                               align_corners=True).permute(0, 2, 3, 1)
        #         PRM_x = PRM_x.view(B, H * W, C)
        #     # PRM_x = PRM_x.view(B, H1, W1, C)
        #     # PRM_x = F.interpolate(PRM_x.permute(0, 3, 1, 2), size=(H, W), mode='bilinear', align_corners=True).permute(0, 2, 3, 1)
        #     # PRM_x = PRM_x.view(B, H * W, C)
        #
        #     convX = self.PCM(shortcut)
        #     x = self.attn.attn(self.attn.norm1(PRM_x))
        #     # convX = convX.permute(0, 2, 3, 1).view(*x.shape).contiguous()
        #     convX = convX.permute(0, 2, 3, 1)
        #     if NUM is True:
        #         convX = F.interpolate(convX.permute(0, 3, 1, 2), size=(H, W), mode='bilinear',
        #                               align_corners=True).permute(
        #             0, 2, 3, 1)
        #     convX = convX.view(*x.shape).contiguous()
        #     x = x + self.attn.drop_path(convX * self.gamma2)
        #     x = x + self.attn.drop_path(self.gamma3 * self.attn.mlp(self.attn.norm2(x)))
        # x = self.SE(x)

        return x



class PromptedNormalCell(NormalCell):
    def __init__(self, num_prompts, prompt_location, deep_prompt, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, class_token=False, group=64, tokens_type='transformer',
                shift_size=0, window_size=0, gamma=False, init_values=1e-4, SE=False, img_size=224, relative_pos=False):
        super(PromptedNormalCell, self).__init__(
            dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop,
            drop_path, act_layer, norm_layer, class_token, group,
            tokens_type,
            shift_size, window_size, gamma, init_values, SE, img_size, relative_pos
        )

        self.num_prompts = num_prompts
        self.prompt_location = prompt_location

        if tokens_type == 'swin':
            if self.prompt_location == "prepend":
                self.attn = PromptedWindowAttention(num_prompts, prompt_location, in_dim=dim, out_dim=dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
                qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, relative_pos=relative_pos)


    def forward(self, x):
        # print(x.shape)

        b, n, c = x.shape
        shortcut = x
        if self.tokens_type == 'swin':
            H, W = self.img_size, self.img_size
            x = self.norm1(x)

            B, L, C = x.shape
            # print(x.shape)
            # print(self.img_size)
            if self.prompt_location == "prepend":
                # change input size
                prompt_emb = x[:, :self.num_prompts, :]
                x = x[:, self.num_prompts:, :]
                L = L - self.num_prompts

            assert L == self.img_size * self.img_size, "input feature has wrong size"

            x = x.view(b, H, W, c)
            # cyclic shift
            if self.shift_size > 0:
                shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            else:
                shifted_x = x
            # partition windows
            x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
            x_windows = x_windows.view(-1, self.window_size * self.window_size, c)  # nW*B, window_size*window_size, C

            num_windows = int(x_windows.shape[0] / B)
            if self.prompt_location == "prepend":
                # expand prompts_embs
                # B, num_prompts, C --> nW*B, num_prompts, C
                prompt_emb = prompt_emb.unsqueeze(0)
                prompt_emb = prompt_emb.expand(num_windows, -1, -1, -1)
                prompt_emb = prompt_emb.reshape((-1, self.num_prompts, C))
                x_windows = torch.cat((prompt_emb, x_windows), dim=1)

            # W-MSA/SW-MSA
            attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

            # seperate prompt embs --> nW*B, num_prompts, C
            if self.prompt_location == "prepend":
                # change input size
                prompt_emb = attn_windows[:, :self.num_prompts, :]
                attn_windows = attn_windows[:, self.num_prompts:, :]
                # change prompt_embs's shape:
                # nW*B, num_prompts, C - B, num_prompts, C
                prompt_emb = prompt_emb.view(-1, B, self.num_prompts, C)
                prompt_emb = prompt_emb.mean(0)

            # merge windows
            attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

            # reverse cyclic shift
            if self.shift_size > 0:
                x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
                x = shifted_x
            x = x.view(b, H * W, c)

            # add the prompt back:
            if self.prompt_location == "prepend":
                x = torch.cat((prompt_emb, x), dim=1)

        else:
            x = self.gamma1 * self.attn(self.norm1(x))

        if self.class_token:

            # B, L, C = x.shape
            # if self.prompt_location == "prepend":
            #     # change input size
            #     prompt_emb = x[:, :self.num_prompts, :]
            #     x = x[:, self.num_prompts:, :]
            #     n = L - self.num_prompts

            n = n - 1
            wh = int(math.sqrt(n))
            convX = self.drop_path(self.gamma2 * self.PCM(shortcut[:, 1:, :].view(b, wh, wh, c).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous().view(b, n, c))
            x = shortcut + self.drop_path(self.gamma1 * x)
            x[:, 1:] = x[:, 1:] + convX
        else:
            # print(shortcut.shape)
            # if self.prompt_location == "prepend":
            #     # change input size
            #     # prompt_emb = x[:, :self.num_prompts, :]
            #     prompt_emb_x = shortcut[:, :self.num_prompts, :]
            #     # x = x[:, self.num_prompts:, :]
            #     shortcut_ci = shortcut[:, self.num_prompts:, :]
            #     n = n - self.num_prompts

            if self.prompt_location == "prepend":
                # change input size
                prompt = shortcut[:, :self.num_prompts, :].unsqueeze(dim=1)
                shortcut_p = shortcut[:, self.num_prompts:, :]
                n = n - self.num_prompts

            # print(shortcut.shape)
            wh = int(math.sqrt(n))

            convX_s = self.drop_path(
                self.gamma2 * self.PCM(shortcut_p.view(b, wh, wh, c).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous().view(b, n, c))
            convX_p = self.drop_path(
                self.gamma2 * self.PCM(prompt.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3,
                                                                                                               1).contiguous().view(
                    b, self.num_prompts, c))
            convX = torch.cat((convX_p, convX_s), dim=1)

            # shortcut_ci = shortcut_ci.view(b, wh, wh, c)
            # prompt_emb_x = prompt_emb_x.unsqueeze(dim=1).expand(b, wh, self.num_prompts, c)
            # # print(prompt_emb_x.shape)
            # # prompt_emb_y = prompt_emb.unsqueeze(dim=2)
            # shortcut_ci = torch.cat((prompt_emb_x, shortcut_ci), dim=2)
            # # shortcut_ci = torch.cat((prompt_emb_y, shortcut_ci), dim=1)
            # shortcut_ci = self.gamma2 * self.PCM(shortcut_ci.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1)
            # shortcut_ci_x = shortcut_ci[:, :, :self.num_prompts, :]
            # shortcut_ci = shortcut_ci[:, :, self.num_prompts:, :]
            # # print(shortcut_ci.shape)
            # convX = self.drop_path(shortcut_ci.contiguous().view(b, n, c))
            # prompt_emb_x = self.drop_path(shortcut_ci_x).mean(dim=1)
            # if self.prompt_location == "prepend":
            #     convX = torch.cat((prompt_emb_x, convX), dim=1)

            # convX = self.drop_path(self.gamma2 * self.PCM(shortcut.view(b, wh, wh, c).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous().view(b, n, c))
            x = shortcut + self.drop_path(self.gamma1 * x) + convX
            # if self.prompt_location == "prepend":
            #     x = torch.cat((prompt_emb, x), dim=1)
            # x = x + convX
        x = x + self.drop_path(self.gamma3 * self.mlp(self.norm2(x)))
        x = self.SE(x)
        # print(x)
        return x


# class PromptedToken_transformer(Token_transformer):
#
#     def __init__(self, num_prompts, prompt_location, dim, in_dim, num_heads, mlp_ratio=1., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
#                  drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, gamma=False, init_values=1e-4):
#         super(PromptedToken_transformer, self).__init__(
#             dim, in_dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop,
#             drop_path, act_layer, norm_layer, gamma, init_values
#         )
#         self.norm1 = norm_layer(dim)
#         self.attn = Attention(
#             dim, in_dim=in_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, gamma=gamma, init_values=init_values)
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.norm2 = norm_layer(in_dim)
#         self.mlp = Mlp(in_features=in_dim, hidden_features=int(in_dim*mlp_ratio), out_features=in_dim, act_layer=act_layer, drop=drop)
#
#     def forward(self, x):
#         x = self.attn(self.norm1(x))
#         x = x + self.drop_path(self.mlp(self.norm2(x)))
#         return x



class PromptedSwinTransformerBlock(SwinTransformerBlock):
    def __init__(
        self, num_prompts, prompt_location, in_dim, out_dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 relative_pos=True, act_layer=nn.GELU, norm_layer=nn.LayerNorm
    ):
        super(PromptedSwinTransformerBlock, self).__init__(
            in_dim, out_dim, input_resolution, num_heads, window_size, shift_size,
            mlp_ratio, qkv_bias, qk_scale, drop, attn_drop, drop_path,
            relative_pos, act_layer, norm_layer)
        self.num_prompts = num_prompts
        self.prompt_location = prompt_location
        if self.prompt_location == "prepend":
            self.attn = PromptedWindowAttention(
                num_prompts, prompt_location,in_dim=in_dim, out_dim=out_dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, relative_pos=relative_pos)

    def forward(self, x):
        # print(x.shape)
        H, W = self.input_resolution
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)
        # print(self.num_prompts)

        if self.prompt_location == "prepend":
            # change input size
            prompt_emb = x[:, :self.num_prompts, :]
            x = x[:, self.num_prompts:, :]
            L = L - self.num_prompts

        assert L == H * W, "input feature has wrong size, should be {}, got {}".format(H*W, L)

        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows --> nW*B, window_size, window_size, C
        x_windows = window_partition(shifted_x, self.window_size)
        # nW*B, window_size*window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        # nW*B, window_size*window_size, C

        # add back the prompt for attn for parralel-based prompts
        # nW*B, num_prompts + window_size*window_size, C
        num_windows = int(x_windows.shape[0] / B)
        if self.prompt_location == "prepend":
            # expand prompts_embs
            # B, num_prompts, C --> nW*B, num_prompts, C
            prompt_emb = prompt_emb.unsqueeze(0)
            prompt_emb = prompt_emb.expand(num_windows, -1, -1, -1)
            prompt_emb = prompt_emb.reshape((-1, self.num_prompts, C))
            x_windows = torch.cat((prompt_emb, x_windows), dim=1)

        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # seperate prompt embs --> nW*B, num_prompts, C
        if self.prompt_location == "prepend":
            # change input size
            prompt_emb = attn_windows[:, :self.num_prompts, :]
            attn_windows = attn_windows[:, self.num_prompts:, :]
            # change prompt_embs's shape:
            # nW*B, num_prompts, C - B, num_prompts, C
            prompt_emb = prompt_emb.view(-1, B, self.num_prompts, C)
            prompt_emb = prompt_emb.mean(0)

        # merge windows
        attn_windows = attn_windows.view(
            -1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(
            attn_windows, self.window_size, H, W)  # B H W C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2)
            )
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # add the prompt back:
        if self.prompt_location == "prepend":
            x = torch.cat((prompt_emb, x), dim=1)
        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PromptedWindowAttention(WindowAttention):
    def __init__(
        self, num_prompts, prompt_location, in_dim, out_dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., relative_pos=False
    ):
        super(PromptedWindowAttention, self).__init__(
            in_dim, out_dim, window_size, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop, relative_pos)
        self.num_prompts = num_prompts
        self.prompt_location = prompt_location

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if self.relative_pos:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1],
                -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            # account for prompt nums for relative_position_bias
            # attn: [1920, 6, 649, 649]
            # relative_position_bias: [6, 49, 49])

            if self.prompt_location == "prepend":
                # expand relative_position_bias
                _C, _H, _W = relative_position_bias.shape

                relative_position_bias = torch.cat((
                    torch.zeros(_C, self.num_prompts, _W, device=attn.device),
                    relative_position_bias
                ), dim=1)
                relative_position_bias = torch.cat((
                    torch.zeros(_C, _H + self.num_prompts, self.num_prompts, device=attn.device),
                    relative_position_bias
                ), dim=-1)

            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            # incorporate prompt
            # mask: (nW, 49, 49) --> (nW, 49 + n_prompts, 49 + n_prompts)
            nW = mask.shape[0]
            if self.prompt_location == "prepend":
                # expand relative_position_bias
                mask = torch.cat((
                    torch.zeros(nW, self.num_prompts, _W, device=attn.device),
                    mask), dim=1)
                mask = torch.cat((
                    torch.zeros(
                        nW, _H + self.num_prompts, self.num_prompts,
                        device=attn.device),
                    mask), dim=-1)
            # logger.info("before", attn.shape)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            # logger.info("after", attn.shape)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x