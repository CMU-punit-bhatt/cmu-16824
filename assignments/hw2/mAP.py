import torch

from utils import iou

def calculate_map(bboxes,
                  scores,
                  preds,
                  gt_boxes,
                  gt_classes,
                  iou_thresh=0.3,
                  n_classes=20,
                  eps=1e-5,
                  return_ap=True):
    """_summary_

    Args:
        bboxes (_type_): _description_
        scores (_type_): _description_
        preds (_type_): _description_
        gt_boxes (_type_): _description_
        gt_classes (_type_): _description_
        iou_thresh (float, optional): _description_. Defaults to 0.3.
        n_classes (int, optional): _description_. Defaults to 20.
        eps (_type_, optional): _description_. Defaults to 1e-5.
        return_ap (bool, optional): _description_. Defaults to True.
    """

        # bboxes (_type_): A (M, N, 4) tensor.
        # scores (_type_): A (M, N,) tensor.
        # preds : A (M, N) tensor indicating class.
        # gt_boxes (_type_): List of M (K_i, 4) tensors.
        # gt_classes : List of M (K_i,) tensors
        # iou_thresh (float, optional): _description_. Defaults to 0.3.
        # score_thresh (float, optional): _description_. Defaults to 0.045.

    aps = []

    for m in range(len(bboxes)):
        order = torch.argsort(scores[m], descending=True)
        bboxes[m] = bboxes[m][order]

    for c in range(n_classes):

        tp = 0
        fp = 0
        tps = []
        fps = []
        gts_count = 0.

        for m in range(len(bboxes)):

            bboxes_m = bboxes[m][preds[m] == c]
            gts_m = gt_boxes[m][gt_classes[m] == c]
            visited = torch.zeros(gts_m.size(0))
            total_visited = 0.

            gts_count += int(gts_m.size(0))

            for l in range(bboxes_m.size(0)):

                best_iou = 0.
                best_iou_idex = None

                for k in range(gts_m.size(0)):

                    if visited[k] == 1:
                        continue

                    iou_k = iou(bboxes_m[l], gts_m[k])

                    if iou_k > best_iou:
                        best_iou = iou_k
                        best_iou_idex = k

                if best_iou >= iou_thresh:
                    tp += 1.
                    visited[best_iou_idex] = 1
                    total_visited += 1

                else:
                    fp += 1

                tps.append(tp)
                fps.append(fp)

                if gts_m.size(0) == total_visited:
                    break

        # Calculating precision and recall.
        tps = torch.Tensor(tps)
        fps = torch.Tensor(fps)

        precisions = tps / (tps + fps + eps)
        recalls = tps / (gts_count + eps)

        aps.append(torch.trapz(precisions, recalls))

    aps = torch.Tensor(aps)

    if return_ap:
        return torch.mean(aps), aps

    return torch.mean(aps)

