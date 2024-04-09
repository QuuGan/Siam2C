# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------


# --------------------------------------------------------
# Mask
# --------------------------------------------------------


import torch.nn as nn


class Mask(nn.Module):
    # mask的基本信息
    def __init__(self):
        # 初始化
        super(Mask, self).__init__()

    def forward(self, z_f, x_f):
        # 向前传播
        raise NotImplementedError

    def template(self, template):
        # 模板信息
        raise NotImplementedError

    def track(self, search):
        # 跟踪信息
        raise NotImplementedError

    def param_groups(self, start_lr, feature_mult=1):
        # 过滤掉不符合条件的元素
        params = filter(lambda x:x.requires_grad, self.parameters())
        params = [{'params': params, 'lr': start_lr * feature_mult}]
        return params
