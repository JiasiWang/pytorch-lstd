# -*- coding: utf-8 -*-
import torch
import cv2

def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def bbox_overlaps(boxes,query_boxes):
    """
    Parameters
    ----------
    boxes: (N, 4) ndarray of float
    query_boxes: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """

    N = boxes.size(0)
    K = query_boxes.size(0)
    overlaps = torch.FloatTensor(N, K)
    overlaps[:] = 0
 
    for k in range(K):
        box_area = (
            (query_boxes[k, 2]*300 - query_boxes[k, 0]*300 + 1) *
            (query_boxes[k, 3]*300 - query_boxes[k, 1]*300 + 1)
        )
        for n in range(N):
            iw = (
                min(boxes[n, 2]*300, query_boxes[k, 2]*300) -
                max(boxes[n, 0]*300, query_boxes[k, 0]*300) + 1
            )
            if iw > 0:
                ih = (
                    min(boxes[n, 3]*300, query_boxes[k, 3]*300) -
                    max(boxes[n, 1]*300, query_boxes[k, 1]*300) + 1
                )
                if ih > 0:
                    ua = float(
                        (boxes[n, 2]*300 - boxes[n, 0]*300 + 1) *
                        (boxes[n, 3]*300 - boxes[n, 1]*300 + 1) +
                        box_area - iw * ih
                    )
                    overlaps[n, k] = iw * ih / ua
    return overlaps.cuda()


def match_proposal(truths, the_proposal):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        threshold: (float) The overlap threshold used when mathing boxes.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        idx: (int) current batch index
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """

  

    overlaps1 = bbox_overlaps(
        truths,
        the_proposal
    )
    overlaps = jaccard(
        truths,
        the_proposal
    )
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    print(best_prior_idx)
    # [1,num_priors] best ground truth for each prior
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    print(best_truth_idx)
    best_truth_idx.squeeze_(0)
    print(best_truth_idx)
    best_truth_overlap.squeeze_(0)
    print(best_truth_overlap)
    best_prior_idx.squeeze_(1)
    print(best_prior_idx)
    best_prior_overlap.squeeze_(1)
    print(best_prior_overlap)
    best_truth_overlap.index_fill_(0, best_prior_idx, 2) 
    print(best_truth_overlap)

    print(best_truth_idx)
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j
    print(best_truth_idx)
    return overlaps, overlaps1

def clamp(boxes):

    for i in range(boxes.size(0)):
        box = boxes[i]
        for j in range(4):
            if box[j] < 0:
                box[j] *= -1
        if box[2] <= box[0]:
            box[2] += box[0]
        if box[3] <= box[1]:
            box[3] += box[1]

        box[:] /= box.max()
        box[:] *= 300

if __name__ == "__main__":

    truth = torch.randn(3,4)
    the_proposal = torch.randn(15,4)
    clamp(truth)
    clamp(the_proposal)

    print(truth)
    print(the_proposal)

    a,b = match_proposal(truth, the_proposal)

    print(a)
    print('----------------------')
    print(b)