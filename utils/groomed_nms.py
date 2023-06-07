

import os, sys
sys.path.append(os.getcwd())

import numpy as np
import itertools
import torch

# for inference, it can replace the regular nms
def groomed_nms(
        preds,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nm=0,  # number of masks
):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping detections

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    if isinstance(preds, (list, tuple)): # YOLOv5 model in validation model, output = (inference_out, loss_out)
        preds = preds[0] # [64, 18522, 85] batch-size, boxes, class+5
    
    nm = 0
    bs = preds.shape[0]  # batch size
    nc = preds.shape[2] - nm - 5  # number of classes
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    mi = 5 + nc  # mask start index

    output = [torch.zeros((0, 6 + nm), device=preds.device)] * bs
    for xi, aboxes in enumerate(preds):  # image index, image inference
        # Compute conf
        aboxes[:, 5:] *= aboxes[:, 4:5]  # conf = obj_conf * cls_conf
        # Box/Mask
        box = xywh2xyxy(aboxes[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        mask = aboxes[:, mi:]  # zero columns if no masks

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (aboxes[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
            aboxes = torch.cat((box[i], aboxes[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = aboxes[:, 5:mi].max(1, keepdim=True)
            aboxes = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            aboxes = aboxes[(aboxes[:, 5:6] == torch.tensor(classes, device=aboxes.device)).any(1)]

        # first sort
        # sorted_inds = (-aboxes[:, 4]).argsort()
        # original_inds = (sorted_inds).argsort()
        # aboxes = aboxes[sorted_inds, :]
        # print(aboxes.shape)
        # cls_pred = cls_pred[sorted_inds]

        # pre-nms
        nms_topN_pre = 3000
        # cls_pred = cls_pred[0:min(nms_topN_pre, cls_pred.shape[0])]
        # aboxes = aboxes[0:min(nms_topN_pre, aboxes.shape[1]), :]

        aboxes = aboxes[aboxes[:, 4].argsort(descending=True)[:nms_topN_pre]]  # sort by confidence and remove excess boxes

        # nms
        num_boxes = 500
        aboxes = aboxes[:num_boxes, :]
        ious_2d_for_nms_img = iou(aboxes[:, 0:4], aboxes[:, 0:4], mode='combinations')
        ious_for_nms_img    = ious_2d_for_nms_img
        # print("ious_for_nms_img: ", ious_for_nms_img.shape)

        # one photo for every inference
        keep_inds, _, scores_new = differentiable_nms(scores_unsorted= aboxes[:, 4], iou_unsorted= ious_for_nms_img, nms_threshold= 0.6, pruning_method= "linear", temperature= 0.1, valid_box_prob_threshold = 0.1, return_sorted_prob= False, group_boxes= True, mask_group_boxes= True, group_size= 100)
        # keep_inds = keep_inds.numpy()
        # suppress boxes
        aboxes = aboxes[keep_inds, :]
        # print("aboxes after nms: ", aboxes.shape)

        output[xi] = aboxes
    return output

def differentiable_nms(scores_unsorted, iou_unsorted, nms_threshold= 0.4, pruning_method= "linear", temperature= 0.01, valid_box_prob_threshold= 0.3, return_sorted_prob= False, sorting_method= "hard", sorting_temperature= None, group_boxes= True, mask_group_boxes= True, group_size= 100, debug= False):
    """
        GrooMeD-NMS: Grouped Mathematical Differentiable NMS.
        Abhinav Kumar

        :param scores_unsorted:          Unsorted scores of the boxes Tensor (N, )
        :param iou_unsorted:             Overlap matrix of the boxes  Tensor (N, N)
        :param nms_threshold:            NMS threshold
        :param pruning_method:           Pruning method/function- Can be one of linear/soft_nms (exponential)/sigmoidal
        :param temperature:              Temperature of the pruning function. If pruning function is linear, this is not used
        :param valid_box_prob_threshold: Used to decide if the box is valid after re-scoring.
        :param return_sorted_prob:       Whether should we return sorted scores or not
        :param sorting_method:           Sorting should be hard or soft
        :param sorting_temperature:      Sorting temperature for soft sorting. If not supplied, sorting_temperature = temperature
        :param group_boxes:              Grouping should be carried out or not
        :param mask_group_boxes:         Masking should be carried out or not
        :param group_size:               Maximum Group size of the group
        :param debug:                    Only for debugging
        :return:
               valid_boxes_index:        Valid box indices after NMS   (Variable tensor of shape between 1  and N)
               invalid_boxes_index:      Invalid box indices after NMS (Variable tensor of shape between N-1 and 0)
               non_suppression_prob:     Re-scores or updated-scores of the boxes after NMS Tensor (N, )
    """

    if type(scores_unsorted) == np.ndarray:
        scores_unsorted = torch.from_numpy(scores_unsorted).float()
        iou_unsorted    = torch.from_numpy(iou_unsorted).float()

    #======================================================================
    # Do the sorting of the scores and the corresponding IoU matrix
    #======================================================================
    _, indices = torch.sort(scores_unsorted, descending=True)
    if sorting_method == "soft":
        if sorting_temperature is None:
            sorting_temperature = temperature
        scores, convex_comb_matrix, iou = soft_sort(scores_unsorted, full_matrix= iou_unsorted, temperature= sorting_temperature)
    else:
        scores = scores_unsorted[indices]
        iou = iou_unsorted[indices][:, indices]

    if debug:
        print("\nInside diff NMS... After sorting")
        print(scores)
        print(iou)

    num_boxes = scores.shape[0]
    mask      = torch.eye(num_boxes).byte()
    identity  = torch.eye(num_boxes)
    ones      = torch.ones(num_boxes,)

    mask      = cast_to_cpu_cuda_tensor(mask, iou)
    identity  = cast_to_cpu_cuda_tensor(identity, iou)
    ones      = cast_to_cpu_cuda_tensor(ones, iou)

    if  group_boxes:
        inversion_matrix         = torch.zeros(iou.shape)
        inversion_matrix         = cast_to_cpu_cuda_tensor(inversion_matrix, iou)

    #======================================================================
    # Prune Matrix
    #======================================================================
    overlap_probabilities        = pruning_function(iou, nms_threshold= nms_threshold, temperature= temperature, pruning_method= pruning_method)
    overlap_probabilities        = torch.tril(overlap_probabilities)
    overlap_probabilities.masked_fill_(mask, 0)

    phi_matrix                   = overlap_probabilities

    #======================================================================
    # Inversion or Subtraction
    #======================================================================
    if group_boxes:

        #==================================================================
        # Grouping
        #==================================================================
        groups = get_groups(iou_unsorted= iou, group_threshold= nms_threshold, group_size= group_size, scores_unsorted= scores)

        if debug:
            print("Groups")
            print(groups)

        num_groups = len(groups)
        #==================================================================
        # Masking
        #==================================================================
        if mask_group_boxes:
            mask = mask.float()
            mask[:,:]  = 0
            for j in range(num_groups):
                mask[groups[j], groups[j][0]] = 1
            phi_matrix = phi_matrix * mask

        # Do groupwise inversion
        for j in range(num_groups):
            if mask_group_boxes:
                temp_store = identity[groups[j]][:, groups[j]] - phi_matrix[groups[j]][:, groups[j]]
            else:
                temp_store = torch.inverse(identity[groups[j]][:, groups[j]] + phi_matrix[groups[j]][:, groups[j]])
            inversion_matrix         = indices_copy(A= inversion_matrix, B= temp_store, indA= groups[j])
    else:
        inversion_matrix             = torch.inverse(identity + phi_matrix)
    non_suppression_prob_unsort  = torch.clamp( torch.matmul(inversion_matrix, scores), min= 0, max= 1)
    # non_suppression_prob_unsort  = torch.min(non_suppression_prob_unsort, scores)

    non_suppression_prob_unsort_2 = non_suppression_prob_unsort.clone()
    non_suppression_prob_unsort[non_suppression_prob_unsort < valid_box_prob_threshold] = 0
    if return_sorted_prob:
        non_suppression_prob, sorted_indices = torch.sort(non_suppression_prob_unsort, descending= True)
        valid_boxes_index     = indices[sorted_indices[non_suppression_prob >= valid_box_prob_threshold]]
        invalid_boxes_index   = indices[sorted_indices[non_suppression_prob <  valid_box_prob_threshold]]
    else:
        dummy, sorted_indices = torch.sort(non_suppression_prob_unsort, descending= True)
        valid_boxes_index     = indices[sorted_indices[dummy >= valid_box_prob_threshold]]
        invalid_boxes_index   = indices[sorted_indices[dummy < valid_box_prob_threshold]]
        if group_boxes:
            non_suppression_prob  = non_suppression_prob_unsort_2
        else:
            non_suppression_prob  = non_suppression_prob_unsort

    return valid_boxes_index, invalid_boxes_index, non_suppression_prob

def soft_sort(scores, full_matrix= None, temperature= 0.01):
    """
        Soft sorting of a vector or a matrix. Returns the soft sorted scores.
        Also returns soft sorted full matrix sorted by scores if full_matrix
        is provided.
        Taken from SoftSort: A Continuous Relaxation for the argsort Operator
        Prillo et al, ICML 2020
        https://arxiv.org/pdf/2006.16038.pdf

        Inputs-
        :param scores:      Unsorted scores of the boxes Tensor (N, )
        :param full_matrix: Another tensor which is sorted based on permutation vector of scores.
        :param temperature: Soft-sorting temperature
        :return:
    """
    hard_sorted_scores, _  = torch.sort(scores, descending= True)
    # Small tau can blow up the numerical values. Use numerically stable
    # softmax by first subtracting the max values from the individual heatmap
    # Use https://stackoverflow.com/a/49212689
    init_comb_matrix       = -torch.abs(scores - hard_sorted_scores.unsqueeze(1))
    max_m                  = torch.max(init_comb_matrix, dim= 1)[0]
    max_m_matrix           = max_m.unsqueeze(-1).expand(-1, scores.shape[0])
    convex_comb_matrix     = (init_comb_matrix - max_m_matrix)
    convex_comb_matrix     = torch.exp(convex_comb_matrix/temperature)
    sum_convex_comb_matrix = torch.sum(convex_comb_matrix, dim= 1) + 1e-3
    convex_comb_matrix     = convex_comb_matrix/sum_convex_comb_matrix
    # convex_comb_matrix     = torch.nn.functional.softmax(-torch.abs(scores - hard_sorted_scores.unsqueeze(1))/temperature, dim= 1)

    soft_sorted_scores     = torch.matmul(convex_comb_matrix, scores)

    if full_matrix is None:
        return soft_sorted_scores, convex_comb_matrix
    else:
        soft_sorted_matrix = torch.matmul(convex_comb_matrix, full_matrix)
        return soft_sorted_scores, convex_comb_matrix, soft_sorted_matrix

def pruning_function(iou, nms_threshold= 0.4, temperature= 0.01, pruning_method= "linear"):

    if type(iou) == torch.Tensor:
        if pruning_method == "sigmoidal":
            overlap_probabilities = torch.sigmoid((iou - nms_threshold) / temperature)
        elif pruning_method == "linear":
            overlap_probabilities = iou
        elif pruning_method == "soft_nms":
            overlap_probabilities = 1- torch.exp(-torch.pow(iou, 2)/temperature)
        else:
            raise NotImplementedError("Pruning method not implemented!")

    elif type(iou) == np.ndarray:
        if pruning_method == "sigmoidal":
            overlap_probabilities = sigmoid_numpy((iou - nms_threshold) / temperature)
        elif pruning_method == "linear":
            overlap_probabilities = iou
        elif pruning_method == "soft_nms":
            overlap_probabilities = 1- np.exp(-np.power(iou, 2)/temperature)
        else:
            raise NotImplementedError("Pruning method not implemented!")

    return overlap_probabilities

def sigmoid_numpy(x):
    y = np.zeros(x.shape)
    index_gt = np.where(x>0)[0]
    y[index_gt] = 1.0/(1+np.exp(-x[index_gt]))

    index_lt = np.where(x<=0)[0]
    y[index_lt] = np.exp(x[index_lt])/(1+np.exp(x[index_lt]))

    return y

def cast_to_cpu_cuda_tensor(input, reference_tensor):
    if reference_tensor.is_cuda and not input.is_cuda:
        input = input.cuda()
    if not reference_tensor.is_cuda and input.is_cuda:
        input = input.cpu()
    return input

def get_groups(iou_unsorted, group_threshold, scores_unsorted, group_size= 100, return_original_indices= True):
    """
        Grouping of the boxes
    """
    # Do the sorting of the scores and the corresponding ious
    scores, indices = torch.sort(scores_unsorted, descending=True)
    iou             = iou_unsorted[indices][:, indices]
    num_boxes       = scores.shape[0]

    """
    # Old code
    # First box is the first group
    groups = [torch.Tensor([0]).long()]
    groups[0] = cast_to_cpu_cuda_tensor(groups[0], iou)

    # Assign remaining boxes to groups
    for i in range(1, num_boxes):
        this_box_index = torch.Tensor([i]).long()
        this_box_index = cast_to_cpu_cuda_tensor(this_box_index, iou)

        num_groups     = len(groups)
        found = False

        for j in range(num_groups):
            if iou[i][groups[j][0]] > group_threshold:
                found = True
                # One group can have a maximum of these many boxes
                if groups[j].shape[0] <= 100:
                    groups[j] = torch.cat((groups[j], this_box_index), 0)
                break
        if not found:
            groups.append(this_box_index)
    """

    groups = []
    shrinking_array = torch.arange(num_boxes).long()
    shrinking_array = cast_to_cpu_cuda_tensor(shrinking_array, iou)
    shrinking_iou   = iou.clone()

    while shrinking_array.nelement() > 0:
        # find which indices have high and low overlap with top_index
        shrinking_high_overlap_index = shrinking_iou[:,0] >  group_threshold
        shrinking_low_overlap_index  = shrinking_iou[:,0] <= group_threshold

        # Add high overlap indices to group
        full_high_overlap_index = shrinking_array[shrinking_high_overlap_index].clone()
        num_elements = min(full_high_overlap_index.shape[0], group_size+1)
        groups.append(full_high_overlap_index[:num_elements].long())

        # Only keep low overlap indices in both shrinking_array and shrinking_iou
        if torch.sum(shrinking_low_overlap_index).item() == 0:
            # Tensor deletion does not work very well when there are zero items to delete
            break
        shrinking_array = shrinking_array[shrinking_low_overlap_index]
        shrinking_iou   = shrinking_iou[shrinking_low_overlap_index][:,shrinking_low_overlap_index]

    num_groups = len(groups)

    if return_original_indices:
        for j in range(num_groups):
            groups[j] = indices[groups[j]]

    return groups

def indices_copy(A, B, indA, indB= None, inplace= True):
    """
        Modified version of
        https://discuss.pytorch.org/t/tensor-index-to-index-copy/57279/2
    """
    # To make sure our views are valid
    if not A.is_contiguous():
        A = A.contiguous()
    if not B.is_contiguous():
        B = B.contiguous()

    # Get the original shape and dimensions
    shapeA = A.shape
    dimA = A.dim()
    dimB = B.dim()

    # Add a new dimension if dimA or dimB is 2
    if dimA == 2:
        A = A.unsqueeze(2)
        new_dimA = dimA + 1
    else:
        new_dimA = dimA

    if dimB == 2:
        B = B.unsqueeze(2)
        new_dimB = dimB + 1
    else:
        new_dimB = dimB

    if indA.dim() == 1:
        indA_clone = indA.clone()
        if indA.is_cuda:
            indA_clone = indA_clone.cpu()
        indA_0_np = indA_clone.numpy()
        indA = torch.Tensor(list(itertools.product(indA_0_np, indA_0_np))).long()
        indA = cast_to_cpu_cuda_tensor(input= indA, reference_tensor= A)

    if indB is None:
        # Get all combinations of B
        indB_0_np = np.arange(B.shape[0])
        indB_1_np = np.arange(B.shape[1])
        indB = torch.Tensor(list(itertools.product(indB_0_np, indB_1_np))).long()
        indB = cast_to_cpu_cuda_tensor(input= indB, reference_tensor= A)

    new_shapeA = A.shape
    new_shapeB = B.shape

    # Collapse the first two dimensions, so that we index only one
    vA = A.view((-1, new_shapeA[new_dimA-1]))
    vB = B.view((-1, new_shapeA[new_dimB-1]))

    # If we need out of place, clone to get a tensor backed by new memory
    if not inplace:
        vA = vA.clone()

    # Transform the 2D indices into 1D indices in our collapsed dimension
    lin_indA = indA.select(1, 0) * new_shapeA[1] + indA.select(1, 1)
    lin_indB = indB.select(1, 0) * new_shapeB[1] + indB.select(1, 1)

    # Read B and write in A
    vA.index_copy_(0, lin_indA, vB.index_select(0, lin_indB))

    if dimA == 2:
        vA = vA.view(new_shapeA).squeeze(2)

    return vA.view(shapeA)

def iou(box_a, box_b, mode='combinations', data_type=None):
    """
    Computes the amount of Intersection over Union (IoU) between two different sets of boxes.

    Args:
        box_a (nparray): Mx4 boxes, defined by [x1, y1, x2, y2]
        box_a (nparray): Nx4 boxes, defined by [x1, y1, x2, y2]
        mode (str): either 'combinations' or 'list', where combinations will check all combinations of box_a and
                    box_b hence MxN array, and list expects the same size list M == N, hence returns Mx1 array.
        data_type (type): either torch.Tensor or np.ndarray, we automatically determine otherwise
    """

    # determine type
    if data_type is None: data_type = type(box_a)

    # this mode computes the IoU in the sense of combinations.
    # i.e., box_a = M x 4, box_b = N x 4 then the output is M x N
    if mode == 'combinations':
        inter = intersect(box_a, box_b, data_type=data_type)  # (500, 500)
        area_a = ((box_a[:, 2] - box_a[:, 0]) *
                  (box_a[:, 3] - box_a[:, 1]))  # (500)
        area_b = ((box_b[:, 2] - box_b[:, 0]) *
                  (box_b[:, 3] - box_b[:, 1]))

        # torch.Tensor
        if data_type == torch.Tensor:
            union = area_a.unsqueeze(0) + area_b.unsqueeze(1) - inter
            return (inter / union).permute(1, 0)

        # np.ndarray
        elif data_type == np.ndarray:
            union = np.expand_dims(area_a, 0) + np.expand_dims(area_b, 1) - inter
            return (inter / union).T

        # unknown type
        else:
            raise ValueError('unknown data type {}'.format(data_type))

    # this mode compares every box in box_a with target in box_b
    # i.e., box_a = M x 4 and box_b = M x 4 then output is M x 1
    elif mode == 'list':

        inter = intersect(box_a, box_b, mode=mode)
        area_a = (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])
        area_b = (box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])
        union = area_a + area_b - inter

        return inter / union

    else:
        raise ValueError('unknown mode {}'.format(mode))

def intersect(box_a, box_b, mode='combinations', data_type=None):
    """
    Computes the amount of intersect between two different sets of boxes.

    Args:
        box_a (nparray): Mx4 boxes, defined by [x1, y1, x2, y2]
        box_a (nparray): Nx4 boxes, defined by [x1, y1, x2, y2]
        mode (str): either 'combinations' or 'list', where combinations will check all combinations of box_a and
                    box_b hence MxN array, and list expects the same size list M == N, hence returns Mx1 array.
        data_type (type): either torch.Tensor or np.ndarray, we automatically determine otherwise
    """

    # determine type
    if data_type is None: data_type = type(box_a)

    # this mode computes the intersect in the sense of combinations.
    # i.e., box_a = M x 4, box_b = N x 4 then the output is M x N
    if mode == 'combinations':

        # np.ndarray
        if data_type == np.ndarray:
            # print("Using combinations in intersect function on numpy")
            # Calculates the coordinates of the overlap box
            # eg if the box x-coords is at 4 and 5, then the overlap will be minimum
            # of the two which is 4
            # np.maximum is to take two arrays and compute their element-wise maximum.
            # Here, 'compatible' means that one array can be broadcast to the other.
            max_xy = np.minimum(box_a[:, 2:4], np.expand_dims(box_b[:, 2:4], axis=1))
            min_xy = np.maximum(box_a[:, 0:2], np.expand_dims(box_b[:, 0:2], axis=1))
            inter = np.clip((max_xy - min_xy), a_min=0, a_max=None)

        elif data_type == torch.Tensor:
            max_xy = torch.min(box_a[:, 2:4], box_b[:, 2:4].unsqueeze(1))
            min_xy = torch.max(box_a[:, 0:2], box_b[:, 0:2].unsqueeze(1))
            inter = torch.clamp((max_xy - min_xy), 0)
            # print(box_a[0][..., 2:4].shape)
            # print(box_b[0][..., 2:4].unsqueeze(1).shape)

        # unknown type
        else:
            raise ValueError('type {} is not implemented'.format(data_type))

        return inter[:, :, 0] * inter[:, :, 1]

    # this mode computes the intersect in the sense of list_a vs. list_b.
    # i.e., box_a = M x 4, box_b = M x 4 then the output is Mx1
    elif mode == 'list':

        # torch.Tesnor
        if data_type == torch.Tensor:
            max_xy = torch.min(box_a[:, 2:], box_b[:, 2:])
            min_xy = torch.max(box_a[:, :2], box_b[:, :2])
            inter = torch.clamp((max_xy - min_xy), 0)

        # np.ndarray
        elif data_type == np.ndarray:
            max_xy = np.minimum(box_a[:, 2:], box_b[:, 2:])
            min_xy = np.maximum(box_a[:, :2], box_b[:, :2])
            inter = np.clip((max_xy - min_xy), a_min=0, a_max=None)

        # unknown type
        else:
            raise ValueError('unknown data type {}'.format(data_type))

        return inter[:, 0] * inter[:, 1]

    else:
        raise ValueError('unknown mode {}'.format(mode))

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

