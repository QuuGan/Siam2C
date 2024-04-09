# --------------------------------------------------------
'''
The current code is based on SiamMask (@inproceedings{wang2019fast,
    title={Fast online object tracking and segmentation: A unifying approach},
    author={Wang, Qiang and Zhang, Li and Bertinetto, Luca and Hu, Weiming and Torr, Philip HS},
    booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
    year={2019}}).
and RBO(@inproceedings{Tang_2022_Ranking,
    title={Ranking-Based Siamese Visual Tracking},
    author={Feng, Tang and Qiang Ling},
    booktitle={The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2022}
})

We have made some notes in Chinese, which we hope you will find helpful if needed.

'''
# --------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.anchors import Anchors
import numpy as np


class SiamMask(nn.Module):
    def __init__(self, anchors=None, o_sz=127, g_sz=127):
        super(SiamMask, self).__init__()
        self.anchors = anchors  # anchor_cfg
        self.anchor_num = len(self.anchors["ratios"]) * len(self.anchors["scales"])
        self.anchor = Anchors(anchors)
        self.features = None
        self.rpn_model = None
        self.mask_model = None
        self.o_sz = o_sz
        self.g_sz = g_sz
        self.upSample = nn.UpsamplingBilinear2d(size=[g_sz, g_sz])

        self.all_anchors = None

    def set_all_anchors(self, image_center, size):
        # cx,cy,w,h
        if not self.anchor.generate_all_anchors(image_center, size):
            return
        all_anchors = self.anchor.all_anchors[1]  # cx, cy, w, h
        self.all_anchors = torch.from_numpy(all_anchors).float().cuda()
        self.all_anchors = [self.all_anchors[i] for i in range(4)]

    def feature_extractor(self, x):
        return self.features(x)

    def rpn(self, template, search):
        pred_cls, pred_loc = self.rpn_model(template, search)
        return pred_cls, pred_loc

    def mask(self, template, search):
        pred_mask = self.mask_model(template, search)
        return pred_mask

    def _add_rpn_loss(self, label_cls, label_loc, lable_loc_weight, label_mask, label_mask_weight,
                      rpn_pred_cls, rpn_pred_loc, rpn_pred_mask):
        rpn_loss_cls = select_cross_entropy_loss(rpn_pred_cls, label_cls)
        rank_cls_loss = Rank_CLS_Loss()
        CR_loss = rank_cls_loss(rpn_pred_cls, label_cls)
        rpn_loss_loc = weight_l1_loss(rpn_pred_loc, label_loc, lable_loc_weight)

        rpn_loss_mask, iou_m, iou_5, iou_7 = select_mask_logistic_loss(rpn_pred_mask, label_mask, label_mask_weight)

        return rpn_loss_cls, CR_loss, rpn_loss_loc, rpn_loss_mask, iou_m, iou_5, iou_7

    def run(self, template, search, softmax=False):
        """
        run network
        """
        template_feature = self.feature_extractor(template)
        feature, search_feature = self.features.forward_all(search)
        rpn_pred_cls, rpn_pred_loc = self.rpn(template_feature, search_feature)
        corr_feature = self.mask_model.mask.forward_corr(template_feature, search_feature)  # (b, 256, w, h)
        rpn_pred_mask = self.refine_model(feature, corr_feature)

        if softmax:
            rpn_pred_cls = self.softmax(rpn_pred_cls)
        return rpn_pred_cls, rpn_pred_loc, rpn_pred_mask, template_feature, search_feature

    def softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2//2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls

    def forward(self, input):
        """
        :param input: dict of input with keys of:
                'template': [b, 3, h1, w1], input template image.
                'search': [b, 3, h2, w2], input search image.
                'label_cls':[b, max_num_gts, 5] or None(self.training==False),
                                     each gt contains x1,y1,x2,y2,class.
        :return: dict of loss, predict, accuracy
        """
        template = input['template']
        search = input['search']
        if self.training:
            label_cls = input['label_cls']
            label_loc = input['label_loc']
            lable_loc_weight = input['label_loc_weight']
            label_mask = input['label_mask']
            label_mask_weight = input['label_mask_weight']

        rpn_pred_cls, rpn_pred_loc, rpn_pred_mask, template_feature, search_feature = \
            self.run(template, search, softmax=self.training)

        outputs = dict()

        outputs['predict'] = [rpn_pred_loc, rpn_pred_cls, rpn_pred_mask, template_feature, search_feature]

        if self.training:
            rpn_loss_cls, CR_loss, rpn_loss_loc, rpn_loss_mask, iou_acc_mean, iou_acc_5, iou_acc_7 = \
                self._add_rpn_loss(label_cls, label_loc, lable_loc_weight, label_mask, label_mask_weight,
                                   rpn_pred_cls, rpn_pred_loc, rpn_pred_mask)
            outputs['losses'] = [rpn_loss_cls, CR_loss, rpn_loss_loc, rpn_loss_mask]
            outputs['accuracy'] = [iou_acc_mean, iou_acc_5, iou_acc_7]

        return outputs

    def template(self, z):
        self.zf = self.feature_extractor(z)
        cls_kernel, loc_kernel = self.rpn_model.template(self.zf)
        return cls_kernel, loc_kernel

    def track(self, x, cls_kernel=None, loc_kernel=None, softmax=False):
        xf = self.feature_extractor(x)
        rpn_pred_cls, rpn_pred_loc = self.rpn_model.track(xf, cls_kernel, loc_kernel)
        if softmax:
            rpn_pred_cls = self.softmax(rpn_pred_cls)
        return rpn_pred_cls, rpn_pred_loc


def get_cls_loss(pred, label, select):
    if select.nelement() == 0: return pred.sum()*0.
    pred = torch.index_select(pred, 0, select)
    label = torch.index_select(label, 0, select)

    return F.nll_loss(pred, label)


def select_cross_entropy_loss(pred, label):
    pred = pred.view(-1, 2)
    label = label.view(-1)
    pos = Variable(label.data.eq(1).nonzero().squeeze()).cuda()
    neg = Variable(label.data.eq(0).nonzero().squeeze()).cuda()

    loss_pos = get_cls_loss(pred, label, pos)
    loss_neg = get_cls_loss(pred, label, neg)
    return loss_pos * 0.5 + loss_neg * 0.5


def weight_l1_loss(pred_loc, label_loc, loss_weight):
    """
    :param pred_loc: [b, 4k, h, w]
    :param label_loc: [b, 4k, h, w]
    :param loss_weight:  [b, k, h, w]
    :return: loc loss value
    """
    b, _, sh, sw = pred_loc.size()
    pred_loc = pred_loc.view(b, 4, -1, sh, sw)
    diff = (pred_loc - label_loc).abs()
    diff = diff.sum(dim=1).view(b, -1, sh, sw)
    loss = diff * loss_weight
    return loss.sum().div(b)


def select_mask_logistic_loss(p_m, mask, weight, o_sz=63, g_sz=127):
    weight = weight.view(-1)
    pos = Variable(weight.data.eq(1).nonzero().squeeze())
    if pos.nelement() == 0: return p_m.sum() * 0, p_m.sum() * 0, p_m.sum() * 0, p_m.sum() * 0

    if len(p_m.shape) == 4:
        p_m = p_m.permute(0, 2, 3, 1).contiguous().view(-1, 1, o_sz, o_sz)
        p_m = torch.index_select(p_m, 0, pos)
        p_m = nn.UpsamplingBilinear2d(size=[g_sz, g_sz])(p_m)
        p_m = p_m.view(-1, g_sz * g_sz)
    else:
        p_m = torch.index_select(p_m, 0, pos)

    mask_uf = F.unfold(mask, (g_sz, g_sz), padding=0, stride=8)
    mask_uf = torch.transpose(mask_uf, 1, 2).contiguous().view(-1, g_sz * g_sz)

    mask_uf = torch.index_select(mask_uf, 0, pos)
    loss = F.soft_margin_loss(p_m, mask_uf)
    iou_m, iou_5, iou_7 = iou_measure(p_m, mask_uf)
    return loss, iou_m, iou_5, iou_7


def iou_measure(pred, label):
    pred = pred.ge(0)
    mask_sum = pred.eq(1).int().add(label.eq(1).int())
    intxn = torch.sum(mask_sum == 2, dim=1).float()
    union = torch.sum(mask_sum > 0, dim=1).float()
    iou = intxn/union
    return torch.mean(iou), (torch.sum(iou > 0.5).float()/iou.shape[0]), (torch.sum(iou > 0.7).float()/iou.shape[0])


class Rank_CLS_Loss(nn.Module):
    def __init__(self, L=4, margin=0.5):
        super(Rank_CLS_Loss, self).__init__()
        # 设置两个参数，margin用来表示负样本的置信度边界
        self.margin =margin
        self.L = L

    def forward(self,input, label):
        # input对应rpn预测的分类结果，即rpn_pred_cls
        # pred对应真实的分类标签
        loss_all = []
        # 获取input第一个维度的大小，表示为batch_size
        batch_size=input.shape[0]

        # 把预测值pred进行维度重塑，三维张量，第一位batch_size大小，第二维自动
        # 第三维2，表示每个数据元素有两个值（预测的类别概率）
        pred=input.view(batch_size,-1,2)
        # 把标签值label进行维度重塑，二维张量
        # 确保真实标签的形状与预测值相匹配，以便后续的损失计算能够进行有效的比较
        label =label.view(batch_size,-1)

        # 对每个样本批次（batch）进行迭代处理，针对批样本计算损失
        for batch_id in range(batch_size):

            # 在当前批次中，找到正样本的索引。这里假设标签为1的为正样本
            pos_index = np.where(label[batch_id].cpu() == 1)[0].tolist()
            # 在当前批次中，找到负样本的索引。这里假设标签为0的为负样本
            neg_index = np.where(label[batch_id].cpu() == 0)[0].tolist()
            
            # 检查是否存在真实正样本，如果存在则执行以下代码，否则跳过当前样本
            if len(pos_index) > 0:

               # 计算正样本对应类别的概率，这里使用了预测值 pred 中正样本对应位置的概率
               pos_prob = torch.exp(pred[batch_id][pos_index][:,1])
               # 计算负样本对应类别的概率，同样使用了预测值 pred 中负样本对应位置的概率
               neg_prob = torch.exp(pred[batch_id][neg_index][:,1])

               # 计算正样本的数量
               num_pos=len(pos_index)

               # 对负样本的概率进行排序
               neg_value, _ = neg_prob.sort(0, descending=True)
               # 对正样本的概率进行排序
               pos_value,_ =pos_prob.sort(0,descending=True)

               # 检查是否存在正样本，如果存在则执行以下代码，否则跳过当前样本
               neg_idx2=neg_prob>0.5
               if neg_idx2.sum()==0:
                   continue

               # 选择与正样本数量相同的前几个负样本
               neg_value=neg_value[0:num_pos]
               # 选择与正样本数量相同的前几个正样本
               pos_value=pos_value[0:num_pos]

               # 计算负样本的 softmax 分布
               neg_q = F.softmax(neg_value, dim=0)
               # 计算负样本的距离，用期望表示，softmax后的权重乘以原始概率值再求和
               neg_dist = torch.sum(neg_value*neg_q)
               # 计算正样本的距离，所有正样本求和再除以正样本个数
               pos_dist = torch.sum(pos_value)/len(pos_value)
               # 计算损失
               loss = torch.log(1.+torch.exp(self.L*(neg_dist - pos_dist+self.margin)))/self.L
            
            # 如果当前批次不存在真实正样本
            else:

               # 找到当前批次中负样本的索引，这里假设标签为0的为负样本
               neg_index = np.where(label[batch_id].cpu() == 0)[0].tolist()
               # 根据负样本的索引，从预测值pred中获取对应位置的概率，并计算负样本的概率值
               neg_prob = torch.exp(pred[batch_id][neg_index][:,1])
               # 将负样本的概率值按降序排序
               neg_value, _ = neg_prob.sort(0, descending=True)

               # 创建一个布尔张量 neg_idx2，用于指示哪些负样本的概率大于0.5，用于筛选出模型较为确信的负样本
               neg_idx2=neg_prob>0.5
               if neg_idx2.sum()==0:
                    continue

               # 计算符合条件的负样本数量，根据阈值条件选取的负样本数量
               num_neg=len(neg_prob[neg_idx2])
               # 将符合条件的负样本数量限制在至少8个。如果选取的负样本数量少于8个，就取8个负样本
               num_neg=max(num_neg,8)
               # 根据负样本数量选取对应数量的负样本概率值
               neg_value=neg_value[0:num_neg]
               # 对选取的负样本概率值进行 softmax 计算，得到负样本的概率分布
               neg_q = F.softmax(neg_value, dim=0)
               # 计算负样本的加权距离（期望），通过将负样本的概率值与 softmax 后的概率分布相乘，然后对结果求和得到
               neg_dist = torch.sum(neg_value*neg_q)
               # 计算损失，正样本直接按1进行计算
               loss = torch.log(1.+torch.exp(self.L*(neg_dist - 1. + self.margin)))/self.L
            loss_all.append(loss)

        if len(loss_all):
            # 列表中的损失值堆叠成一个张量，然后计算张量的均值，得到最终的损失值 final_loss
            final_loss = torch.stack(loss_all).mean()
        else:
            # 如果没有计算到任何损失值，则返回一个零张量
            final_loss=torch.zeros(1).cuda()   
        
        return final_loss   
    

if __name__ == "__main__":