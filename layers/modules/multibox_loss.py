# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from data import v2 as cfg
from ..box_utils import match, match_proposal, log_sum_exp
import cv2
import numpy as np
import matplotlib.pyplot as plt

class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching,
                 bkg_label, neg_mining, neg_pos, neg_overlap, encode_target,
                 use_gpu=True):
        super(MultiBoxLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = cfg['variance']



    def vis(self, img, prior, proposal, truth, labels, cls_t, idx):
        im = img.cpu().data
        im = im.numpy()
        im = im.transpose(1,2,0) 
        img = im[:, :, (2, 1, 0)]
        im = img.copy()
        write_flag = False
        for i in range(truth.size(0)):
            bbox = truth[i,:]
            label = labels[i]
            cv2.rectangle(im, (int(bbox[0]*300), int(bbox[1]*300)),(int(bbox[2]*300), int(bbox[3]*300)),(0,255,0),1)
        '''    
        for j in range(proposal.size(0)):
            write_flag = True
            cv2.rectangle(im, (int(proposal[j][0]*300), int(proposal[j][1]*300)), (int(proposal[j][2]*300), int(proposal[j][3]*300)),(255,255,255),1)
        '''
        for j in range(proposal.size(0)): 
            if cls_t[j]>0:
                write_flag = True
                cv2.rectangle(im, (int(proposal[j][0]*300), int(proposal[j][1]*300)), (int(proposal[j][2]*300), int(proposal[j][3]*300)),(255,0,0),2)   
        
        if write_flag:
            cv2.imwrite('./vis/'+str(idx)+'.jpg', im)
        
    def forward(self, predictions, cls_predictions, proposal, targets, img, im_id, im_vis, h, w):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        loc_data, conf_data, priors = predictions
        num = loc_data.size(0)
        priors = priors[:loc_data.size(1), :]
        num_priors = (priors.size(0))
        num_classes = self.num_classes

        #print('loc_data.shape: ', loc_data.size())                      # 32, 8732, 4
        #print('conf_data.shape: ', conf_data.size())                    # 32, 8732, 2
        #print('priors.shape: ', priors.size())                          #8732, 4
        #print('cls_predictions.shape: ', cls_predictions.size())        #32*100, 21
        #print('rois.shape: ', rois.size())                              # 32*100 , 5

        cls_data = cls_predictions.view(num, -1, 21)
        proposal = proposal.view(num, -1, 5)
       
        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        cls_t = torch.LongTensor(num, 100)   
     
        remove_record = torch.LongTensor(num, 100)
        remove_record[:] = 0
        for idx in range(num):
            truths = targets[idx][:, :-1].data
            labels = targets[idx][:, -1].data
            defaults = priors.data
            the_proposal = proposal[idx].data[:,1:]
            #height, width, channel = im_vis[idx].shape
         
            for i in range(the_proposal.size(0)):
                if the_proposal[i][0]==0 and the_proposal[i][1]==0 and the_proposal[i][2]==0 and the_proposal[i][3]==0:
                    remove_record[idx][i] = 1
            
            match(self.threshold, truths, defaults, self.variance, labels,
                  loc_t, conf_t, idx)
         
            match_proposal(self.threshold, img[idx], truths, the_proposal, labels,
                  cls_t, idx, h, w)

            if 0:
                self.vis(img[idx], defaults, the_proposal, truths, labels, cls_t[idx], im_id[idx][1])
 
        conf_rpn = conf_t.clone()
        row,clo = conf_t.size()
        for row_num in range(row):
            for clo_num in range(clo):
                if conf_rpn[row_num][clo_num] > 0:
                    conf_rpn[row_num][clo_num] = 1

        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
            conf_rpn = conf_rpn.cuda()
            cls_t = cls_t.cuda()
            remove_record = remove_record.cuda()         
        # wrap targets
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)
        conf_rpn = Variable(conf_rpn, requires_grad=False)
        cls_t = Variable(cls_t, requires_grad=False)
        remove_record = Variable(remove_record, requires_grad=False)

        pos = conf_t > 0
        num_pos = pos.sum(dim=1, keepdim=True)

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)

        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_rpn.view(-1, 1))

        # Hard Negative Mining
        loss_c[pos] = 0  # filter out pos boxes for now
        loss_c = loss_c.view(num, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_rpn[(pos+neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        
        N = num_pos.data.sum()
        loss_l /= N
        loss_c /= N
       
        
        #loss for classifier
        cls_pos = cls_t > 0
        cls_num_pos = cls_pos.sum(dim=1, keepdim=True)

        #filter extra proposal cls loss
        #extra = remove_record > 0
           
        # Compute max conf across batch for hard negative mining
        batch_cls = cls_data.view(-1, 21)
        
        loss_cls = log_sum_exp(batch_cls) - batch_cls.gather(1, cls_t.view(-1, 1))

        # Hard Negative Mining
        loss_cls[cls_pos] = 0  # filter out pos boxes for now
        #loss_cls[extra] = 0    #filter extra proposal cls loss        

        loss_cls = loss_cls.view(num, -1)
        _, loss_cls_idx = loss_cls.sort(1, descending=True)
        _, idx_cls_rank = loss_cls_idx.sort(1)
        cls_num_pos = cls_pos.long().sum(1, keepdim=True)
        cls_num_neg = torch.clamp(self.negpos_ratio*cls_num_pos, max=cls_pos.size(1)-1)
        cls_neg = idx_cls_rank < cls_num_neg.expand_as(idx_cls_rank)

        # Confidence Loss Including Positive and Negative Examples
        cls_pos_idx = cls_pos.unsqueeze(2).expand_as(cls_data)
        cls_neg_idx = cls_neg.unsqueeze(2).expand_as(cls_data)
        cls_p = cls_data[(cls_pos_idx+cls_neg_idx).gt(0)].view(-1, 21)
        cls_targets_weighted = cls_t[(cls_pos+cls_neg).gt(0)]
        loss_cls = F.cross_entropy(cls_p, cls_targets_weighted, size_average=False)

        N_cls = cls_num_pos.data.sum()
        loss_cls /= N_cls
        
        return loss_l, loss_c, loss_cls
