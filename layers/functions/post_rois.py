import torch
from torch.autograd import Function
from ..box_utils import decode, nms
from data import v2 as cfg


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

    def forward(self, loc_data, conf_data, prior_data):
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
        num_priors = prior_data.size(0)
        output = torch.zeros(num, self.num_classes-1, self.top_k, 5)
        conf_preds = conf_data.view(num, num_priors,
                                    self.num_classes).transpose(2, 1)

        # Decode predictions into bboxes.
        for i in range(num):
            assert prior_data.cpu().numpy().all() >= 0
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)
           
            # For each class, perform nms
            conf_scores = conf_preds[i].clone()
   

            for cl in range(1, self.num_classes):
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                c_mask[:] = 0

                #filter out boundary boxes
                for n_p in range(decoded_boxes.size(0)):
                    bbox = decoded_boxes[n_p]
                    if bbox.min()<0 or bbox.max()>1:
                        decoded_boxes[:] = 0
                # sort all conf_scores from highest to lowest
                sort_score, sort_index = torch.sort(conf_scores[cl], descending=True)
                sort_index = sort_index[0:1000]
                c_mask[sort_index] = 1
                # get top 1000 boxes
                scores = conf_scores[cl][c_mask]
                scores[:] = i
                if scores.dim() == 0:
                    continue
          
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)
 
                #clip
                #boxes = torch.clamp(boxes, 0, 1)

                # idx of highest scoring and non-overlapping boxes per class
                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
                output[i, (cl-1) , :count] = \
                    torch.cat((scores[ids[:count]].unsqueeze(1),
                               boxes[ids[:count]]), 1)
           
        return output
 
