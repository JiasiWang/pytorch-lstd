import torch
from torch.autograd import Function
from ..box_utils import decode, nms
from data import v2 as cfg
import cv2

class Post_rois(Function):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """
    def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        # Parameters used in nms.
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        self.variance = cfg['variance']

    def forward(self, img, loc_data, conf_data, prior_data):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
        """
        num = loc_data.size(0)  # batch size
        num_priors = prior_data.size(0)  # 8732
        output = torch.zeros(num, self.num_classes-1, self.top_k, 5)
        conf_preds = conf_data.view(num, num_priors,
                                    self.num_classes).transpose(2, 1)

        #print(num)
        #print(prior_data.size()) 8732, 4
        #print(conf_data.size())  8732, 2
        #print(loc_data.size())  

        def vis(img, prior, decoded_boxes, confs, idx):
            im = img.cpu()
            im = im.numpy()
            im = im.transpose(1,2,0)
            img = im[:, :, (2, 1, 0)]
            im = img.copy()
            write_flag = False
            '''
            for j in range(4):
                write_flag = True
                x_c = int(prior[j][0]*300)
                y_c = int(prior[j][1]*300)
                h = int(prior[j][2]*300)
                w = int(prior[j][3]*300)
                cv2.rectangle(im, (x_c - int(w/2), y_c - int(h/2)), (x_c + int(w/2), y_c + int(h/2)),(255,255,255),1)
            '''
            #for i in range(decoded_boxes.size(0)):
            for i in range(10):
                write_flag = True
                cv2.rectangle(im, (int(decoded_boxes[i][0]*300), int(decoded_boxes[i][1]*300)),(int(decoded_boxes[i][2]*300),int(decoded_boxes[i][3]*300)), (255,0,0),1)   
                cv2.putText(im, str(confs[i]), (int(decoded_boxes[i][0]*300) + int((decoded_boxes[i][2]*300 - decoded_boxes[i][0]*300)/2) ,int(decoded_boxes[i][1]*300) + int((decoded_boxes[i][3]*300 - decoded_boxes[i][1]*300)/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255))
            '''
            for j in range(proposal.size(0)):
                if cls_t[j]>0:
                    write_flag = True
                    cv2.rectangle(im, (int(proposal[j][0]*300), int(proposal[j][1]*300)), (int(proposal[j][2]*300), int(proposal[j][3]*300)),(255,0,0),2)
            '''
            if write_flag:
                cv2.imwrite('./vis/'+str(idx)+'.jpg', im)
                #cv2.imshow('./vis/'+str(idx)+'.jpg', im)

        # Decode predictions into bboxes.
        for i in range(num):
            assert prior_data.cpu().numpy().all() >= 0
            prior_data = prior_data.cuda(loc_data[i].get_device())
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)
                
            # For each class, perform nms
            conf_scores = conf_preds[i][1].clone()
            # filter
       
            # apply nms
            ids, count = nms(decoded_boxes, conf_scores, self.nms_thresh, 1000)

            # sort all conf_scores from high to low
            sort_score, sort_index = torch.sort(conf_scores[ids[:count]], descending=True)
            # get top 100
            sort_index = sort_index[0:100]
            scores = conf_scores[ids[:count]][sort_index]
            decoded_boxes = decoded_boxes[ids[:count]][sort_index,:]
            confs = scores.clone()
            # change score to img index
            scores[:] = i
            output[i, 0, :100]= torch.cat((scores.unsqueeze(1), decoded_boxes), 1)
            #output[i, 0, :count]= torch.cat((scores[ids[:count]].unsqueeze(1), decoded_boxes[ids[:count]]), 1)
            '''
            for cl in range(1, self.num_classes):
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                c_mask[:] = 0
                # sort all conf_scores from highest to lowest
                sort_score, sort_index = torch.sort(conf_scores[cl], descending=True)
                sort_index = sort_index[0:1000]
                c_mask[sort_index] = 1
                # get top 1000 boxes
                scores = conf_scores[cl][c_mask]
                if scores.dim() == 0:
                    continue
          
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)
 
                #clip
                #boxes = torch.clamp(boxes, 0, 1)

                # idx of highest scoring and non-overlapping boxes per class
                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
                scores[:] = i
                output[i, (cl-1) , :count] = \
                    torch.cat((scores[ids[:count]].unsqueeze(1),
                               boxes[ids[:count]]), 1)

            '''
            #truths = targets[i][:, :-1].data
            #labels = targets[i][:, -1].data
           
            if 0:
                vis(img[i], prior_data, decoded_boxes, confs, str(i))
               
           
        return output
 
