# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------

# -----------------------------------------------------
# 特征提取基础类
# -----------------------------------------------------

import torch
import torch.nn as nn
import logging

logger = logging.getLogger('global')


class Features(nn.Module):
    def __init__(self):
        super(Features, self).__init__()
        self.feature_size = -1

    def forward(self, x):
        raise NotImplementedError

    def param_groups(self, start_lr, feature_mult=1):
        # 过滤掉不符合条件的元素
        params = filter(lambda x:x.requires_grad, self.parameters())
        params = [{'params': params, 'lr': start_lr * feature_mult}]
        return params

    #模型预训练参数的融合
    def load_model(self, f='pretrain.model'):
        with open(f) as f:
            pretrained_dict = torch.load(f)
            model_dict = self.state_dict()
            print(pretrained_dict.keys())
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            print(pretrained_dict.keys())
            model_dict.update(pretrained_dict)  # 更新
            self.load_state_dict(model_dict)


# 多阶段的特征提取模块，用于custom中的ResDown
class MultiStageFeature(Features):
    def __init__(self):
        super(MultiStageFeature, self).__init__()
        # 初始化
        self.layers = []
        self.train_num = -1
        self.change_point = []
        self.train_nums = []

    def unfix(self, ratio=0.0):
        # 设置在train和test网络层的更新
        if self.train_num == -1:
            self.train_num = 0
            self.unlock()
            self.eval()
        for p, t in reversed(list(zip(self.change_point, self.train_nums))):
            if ratio >= p:
                if self.train_num != t:
                    self.train_num = t
                    self.unlock()
                    return True
                break
        return False

    def train_layers(self):
        # 返回进行参数更新的网络层
        return self.layers[:self.train_num]

    def unlock(self):
        # 设置网络层是否进行梯度更新

        # 将所有参数设置为不可训练
        for p in self.parameters():
            p.requires_grad = False

        logger.info('Current training {} layers:\n\t'.format(self.train_num, self.train_layers()))
        # 将train_layers中的参数，参与梯度更新
        for m in self.train_layers():
            for p in m.parameters():
                p.requires_grad = True

    def train(self, mode):
        self.training = mode
        # 根据mode对网络进行训练
        if mode == False:
            super(MultiStageFeature,self).train(False)
        else:
            for m in self.train_layers():
                m.train(True)

        return self
