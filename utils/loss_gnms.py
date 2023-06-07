# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Loss functions
"""

import torch
import torch.nn as nn
import numpy as np

from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel

from utils.groomed_nms import groomed_nms, differentiable_nms, iou
from utils.aploss import APLoss

def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

# 每次算一张图片的loss
class GnmsLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, preds, pred, true):
        apLoss = APLoss()

        # if self.use_nms_in_loss:
        batch_size, h_times_w_times_anchors, _ = preds.shape  # [64, 18522, 85] batch-size, boxes, class+5
        scores_after_nms  = torch.zeros((batch_size, h_times_w_times_anchors)).type(pred.dtype).cuda()
        targets_after_nms = torch.zeros((batch_size, h_times_w_times_anchors)).type(pred.dtype).cuda()

        labels        = np.zeros(preds.shape[0:2])
        labels_weight = np.zeros(preds.shape[0:2])
        labels_scores = np.zeros(preds.shape[0:2])
        bbox_weights  = np.zeros(preds.shape[0:2])

        batch_size = preds.shape[0]
        for img_index in range(0, batch_size):

            bbox_weights[img_index, fg_inds] = 1
            scores_to_nms_img = torch.max(preds[img_index, :, 1:], dim= 1)[0].clone()

            
            # Sort them in decreasing order
            _, sorted_index = torch.sort(scores_to_nms_img[fg_inds_tensor], descending= True)
            num_boxes_for_nms       = min(500, sorted_index.shape[0])

            # Get the foreground and background for NMS. This will be different from usual foreground since
            # because of the computational issues, we only have at max 500 boxes in the NMS
            # print(fg_inds_tensor)
            fg_index_for_nms        = fg_inds_tensor[sorted_index[:num_boxes_for_nms]]

            if scores_to_nms_img.is_cuda:
                fg_index_np = fg_index_for_nms.cpu()
            else:
                fg_index_np = fg_index_for_nms
            fg_index_np     = fg_index_np.clone().numpy()
            bg_index_for_nms= torch.from_numpy(np.setdiff1d(np.arange(scores_to_nms_img.shape[0]), fg_index_np))


            # Calculate iou2d overlaps
            ious_2d_for_nms_img    = iou(coords_2d_512_img[fg_index_for_nms, :], coords_2d_512_img[fg_index_for_nms, :], mode='combinations')
            
            ious_for_nms_img    = ious_2d_for_nms_img

            # ==============================================================
            # Pass the boxes through our differentiable GrooMeD-NMS
            # ==============================================================
            _, _, scores_after_nms_img = differentiable_nms(scores_unsorted= scores_to_nms_img[fg_index_for_nms], iou_unsorted= ious_for_nms_img.clone().detach(), nms_threshold= self.nms_thres, pruning_method= self.diff_nms_pruning_method, temperature= self.diff_nms_temperature, valid_box_prob_threshold = self.diff_nms_valid_box_prob_threshold, return_sorted_prob= False, group_boxes= self.diff_nms_group_boxes, mask_group_boxes= self.diff_nms_mask_group_boxes, group_size= self.diff_nms_group_size)

            scores_after_nms[img_index, fg_index_for_nms] = scores_after_nms_img

        # Computing loss
        bbox_nms_loss_unweighted = torch.zeros((1,)).float().cuda()
        img_cnt = 0
        for img_index in range(batch_size):
            if np.sum(bbox_weights[img_index]) > 0:
                img_cnt += 1
                bbox_weights_fg_img    = torch.tensor(bbox_weights[img_index], requires_grad=False).type(pred.dtype)
                accept_prob_active_img = bbox_weights_fg_img > 0
                bbox_nms_loss_unweighted += apLoss(scores_after_nms[img_index, accept_prob_active_img], targets_after_nms[img_index, accept_prob_active_img].detach()).squeeze()
        if img_cnt > 0:
            bbox_nms_loss_unweighted /= img_cnt
        bbox_nms_loss_unweighted = bbox_nms_loss_unweighted.squeeze()

        return bbox_nms_loss_unweighted

class ComputeLoss_gnms:
    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, autobalance=False):
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        m = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors
        self.device = device

    def __call__(self, p, targets):  # predictions, targets
        # p: 候选框和三个预测头输出的结果
        # targets: gt box信息，维度是(n, 6)。n是整个batch的图片里gt box的数量，6的每一个维度为(图片在batch中的索引， 目标类别， x, y, w, h)
        if isinstance(p, (list, tuple)): # (torch.cat(z, 1), x)
            preds = p[0] # [64, 18522, 85] batch-size, boxes, class+5
            p = p[1]

        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        lnms = torch.zeros(1, device=self.device)  # after nms loss
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # print(p[0].shape) torch.Size([64, 3, 80, 80, 85])
        # print(targets[0].shape) torch.Size([6])

        # if self.use_nms_loss:
        gnmsloss = GnmsLoss()

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions

                # Regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        # After NMS loss
        # print(tbox[i].shape)
        # lnms += gnmsloss(preds, pbox, tbox[i])

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        lnms *= self.hyp['nms'] # 0.05
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls + lnms) * bs, torch.cat((lbox, lobj, lcls, lnms)).detach()

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=self.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
            ],
            device=self.device).float() * g  # offsets

        for i in range(self.nl):
            anchors, shape = self.anchors[i], p[i].shape
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain  # shape(3,n,7)
            if nt:
                # Matches
                r = t[..., 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices

            # Append
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch
