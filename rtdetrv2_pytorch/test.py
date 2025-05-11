import os 
import time
import torch 
import argparse
import json
import cv2 
import numpy as np 
from tqdm import tqdm
from torchvision.ops import nms
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from src.core import YAMLConfig


def multiclass_nms(
    multi_bboxes, multi_scores, score_thr, nms_cfg, max_num=-1, score_factors=None
):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class), where the last column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS

    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels \
            are 0-based.
    """
    num_classes = multi_scores.size(1) - 1
    # exclude background category
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(multi_scores.size(0), num_classes, 4)
    scores = multi_scores[:, :-1]

    # filter out boxes with low scores
    valid_mask = scores > score_thr

    # We use masked_select for ONNX exporting purpose,
    # which is equivalent to bboxes = bboxes[valid_mask]
    # we have to use this ugly code
    bboxes = torch.masked_select(
        bboxes, torch.stack((valid_mask, valid_mask, valid_mask, valid_mask), -1)
    ).view(-1, 4)
    if score_factors is not None:
        scores = scores * score_factors[:, None]
    scores = torch.masked_select(scores, valid_mask)
    labels = valid_mask.nonzero(as_tuple=False)[:, 1]

    if bboxes.numel() == 0:
        bboxes = multi_bboxes.new_zeros((0, 5))
        labels = multi_bboxes.new_zeros((0,), dtype=torch.long)

        if torch.onnx.is_in_onnx_export():
            raise RuntimeError(
                "[ONNX Error] Can not record NMS "
                "as it has not been executed this time"
            )
        return bboxes, labels

    dets, keep = batched_nms(bboxes, scores, labels, nms_cfg)

    if max_num > 0:
        dets = dets[:max_num]
        keep = keep[:max_num]

    return dets, labels[keep]

def batched_nms(boxes, scores, idxs, nms_cfg, class_agnostic=False):
    """Performs non-maximum suppression in a batched fashion.
    Modified from https://github.com/pytorch/vision/blob
    /505cd6957711af790211896d32b40291bea1bc21/torchvision/ops/boxes.py#L39.
    In order to perform NMS independently per class, we add an offset to all
    the boxes. The offset is dependent only on the class idx, and is large
    enough so that boxes from different classes do not overlap.
    Arguments:
        boxes (torch.Tensor): boxes in shape (N, 4).
        scores (torch.Tensor): scores in shape (N, ).
        idxs (torch.Tensor): each index value correspond to a bbox cluster,
            and NMS will not be applied between elements of different idxs,
            shape (N, ).
        nms_cfg (dict): specify nms type and other parameters like iou_thr.
            Possible keys includes the following.
            - iou_thr (float): IoU threshold used for NMS.
            - split_thr (float): threshold number of boxes. In some cases the
                number of boxes is large (e.g., 200k). To avoid OOM during
                training, the users could set `split_thr` to a small value.
                If the number of boxes is greater than the threshold, it will
                perform NMS on each group of boxes separately and sequentially.
                Defaults to 10000.
        class_agnostic (bool): if true, nms is class agnostic,
            i.e. IoU thresholding happens over all boxes,
            regardless of the predicted class.
    Returns:
        tuple: kept dets and indice.
    """
    nms_cfg_ = nms_cfg.copy()
    class_agnostic = nms_cfg_.pop("class_agnostic", class_agnostic)
    if class_agnostic:
        boxes_for_nms = boxes
    else:
        max_coordinate = boxes.max()
        offsets = idxs.to(boxes) * (max_coordinate + 1)
        boxes_for_nms = boxes + offsets[:, None]
    nms_cfg_.pop("type", "nms")
    split_thr = nms_cfg_.pop("split_thr", 10000)
    if len(boxes_for_nms) < split_thr:
        keep = nms(boxes_for_nms, scores, **nms_cfg_)
        boxes = boxes[keep]
        scores = scores[keep]
    else:
        total_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
        for id in torch.unique(idxs):
            mask = (idxs == id).nonzero(as_tuple=False).view(-1)
            keep = nms(boxes_for_nms[mask], scores[mask], **nms_cfg_)
            total_mask[mask[keep]] = True

        keep = total_mask.nonzero(as_tuple=False).view(-1)
        keep = keep[scores[keep].argsort(descending=True)]
        boxes = boxes[keep]
        scores = scores[keep]

    return torch.cat([boxes, scores[:, None]], -1), keep


class Solver:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda")
        self.model = cfg.model.to(self.device)
        self.optimizer = cfg.optimizer
        self.criterion = cfg.criterion.to(self.device)
        self.postprocessor = cfg.postprocessor
        self.train_dataloader = self.cfg.train_dataloader
        self.eval_dataloader = self.cfg.val_dataloader

    def test(self):
        state = torch.load('output/aug_e12/checkpoint0011.pth', map_location='cpu')
        msg = self.model.load_state_dict(state['model'])
        print(msg)
        self.model.eval()
        out = {}
        for (samples, targets) in tqdm(self.eval_dataloader):
            samples = samples.to(self.device)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            with torch.no_grad():
                outputs = self.model(samples, targets=targets)
            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
            results = self.postprocessor(outputs, orig_target_sizes)
            for t, r in zip(targets, results):
                imgid = int(t['image_id'].cpu().numpy()[0])
                preds = r['boxes'].cpu().numpy().astype(int).tolist()
                scores = r['scores'].cpu().numpy().tolist()
                out[imgid] = {'boxes': preds, 'scores': scores}
        with open('output/aug_e12/results.json', 'w') as f:
            json.dump(out, f)
    
def main():
    cfg = YAMLConfig('configs/rtdetrv2/dronesite.yml')
    solver = Solver(cfg)
    solver.test()

def draw_bbox(image, bboxes, scores=None, inp_type='gt'):
    draw = ImageDraw.Draw(image)
    for bbox, score in zip(bboxes, scores):
        # draw bbox
        if inp_type == 'gt':
            xmin, ymin, w, h = bbox
            xmax = xmin + w
            ymax = ymin + h
            color = (0, 255, 0, 128)
        else:
            xmin, ymin, xmax, ymax = bbox
            color = (255, 0, 0, 128)
        draw.line(
            [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin), (xmin, ymin)],
            width=2,
            fill=color)
        # draw score
        if inp_type == 'det':
            # text = "{} {:.2f}".format(catid2name[catid], score)
            text = f"{score:.1f}"
            left, top, right, bottom = draw.textbbox((0, 0), text)
            tw, th = right - left, bottom - top
            if ymin - th < 0:
                ymin = th
            draw.rectangle(
                [(xmin + 1, ymin - th), (xmin + tw + 1, ymin)], fill=color)
            draw.text((xmin + 1, ymin - th), text, fill=(255, 255, 255))
    return image

def visualize_results():
    results = json.load(open('output/aug_e12/results.json', 'r'))
    annos = json.load(open('/mnt/sda1/datasets/drone_site/dataset/annotations/val_640_025.json', 'r'))
    images = annos['images']
    annotations = annos['annotations']
    imgids = [img['id'] for img in images]
    data = {img['id']: {'file_name': img['file_name']} for img in images}
    for imgid in imgids:
        gts = [ann['bbox'] for ann in annotations if ann['image_id'] == imgid]
        if len(gts) == 0:
            continue
        print(len(gts))
        detections = []
        scores = []
        for bbox, score in zip(results[str(imgid)]['boxes'], results[str(imgid)]['scores']):
            if score < 0.5:
                continue
            detections.append(bbox)
            scores.append(score)
        print(len(detections))
        img = Image.open(os.path.join('/mnt/sda1/datasets/drone_site/dataset/images/val_640_025', data[int(imgid)]['file_name'])).convert('RGB')
        img = draw_bbox(img, detections, scores, 'det')
        
        scores = [1] * len(gts)
        img = draw_bbox(img, gts, scores, 'gt')
        imgname = data[int(imgid)]['file_name']
        img.save(imgname)
        break

def merge_results():
    ori_json = 'output/aug_e12/results.json'
    preds = json.load(open(ori_json, 'r'))
    annos = json.load(open('/mnt/sda1/datasets/drone_site/dataset/annotations/val_640_025.json', 'r'))
    images = annos['images']
    name_id = {img['file_name']: img['id'] for img in images}
    results = {}
    for name, imgid in name_id.items():
        seq_name = '_'.join(name.split('_')[:2])
        if seq_name not in results:
            results[seq_name] = {}
        ori_img_name = '_'.join(name.split('_')[:3]) + '.jpg'
        if ori_img_name not in results[seq_name]:
            results[seq_name][ori_img_name] = {'bboxes':[], 'scores':[]}
        img_xys = name.split('.')[0].split('_')[-4:-2]
        bboxes = np.array(preds[str(imgid)]['boxes'])
        bboxes[:, 0::2] += int(img_xys[0])
        bboxes[:, 1::2] += int(img_xys[1])
        bboxes = bboxes.tolist()
        scores = preds[str(imgid)]['scores']
        results[seq_name][ori_img_name]['bboxes'].extend(bboxes)
        results[seq_name][ori_img_name]['scores'].extend(scores)
    # 开始 nms 合并
    for seq_name, seq_results in results.items():
        for img_name, img_results in seq_results.items():
            bboxes = torch.tensor(img_results['bboxes'], dtype=torch.float32)
            scores = torch.tensor(img_results['scores']).unsqueeze(1)
            padding = scores.new_zeros(scores.shape[0], 1)
            scores = torch.cat([scores, padding], dim=1)
            box, _ = multiclass_nms(
                bboxes,
                scores,
                score_thr=0.5,
                nms_cfg=dict(type="nms", iou_threshold=0.1),
                max_num=600,
            )
            box = box.numpy().tolist()
            print(seq_name, img_name, len(img_results['bboxes']), len(box))
            results[seq_name][img_name]['merged_results'] = box
    with open(ori_json.replace('.json', '_merged.json'), 'w') as f:
        json.dump(results, f)

def visualize_results_global():
    results = json.load(open('output/aug_e12/results_merged.json', 'r'))
    annos = json.load(open('/mnt/sda1/datasets/drone_site/dataset/annotations/val.json', 'r'))
    images = annos['images']
    annotations = annos['annotations']
    img_dict = {img['file_name']: img['id'] for img in images}

    file_name = '/mnt/sda1/datasets/drone_site/dataset/images/val/0001_V_12.jpg'
    img_name = file_name.split('/')[-1]
    seq_name = '_'.join(img_name.split('_')[:2])
    print(seq_name, img_name)
    detections = np.array(results[seq_name][img_name]['merged_results'])
    detections = detections[detections[:, -1] > 0.5, :]
    # ws, hs = detections[:, 2]-detections[:, 0], detections[:, 3]-detections[:, 1]
    # detections = detections[hs/ws > 1, :]
    # detections = detections[hs/ws < 5, :]
    print(len(detections))
    img = Image.open(file_name).convert('RGB')
    img = draw_bbox(img, detections[:, :4], detections[:, -1], 'det')
    # 画gt
    gts = [ann['bbox'] for ann in annotations if ann['image_id'] == img_dict[img_name]]
    scores = [1]*len(gts)
    img = draw_bbox(img, gts, scores, 'gt')
    img.save(os.path.basename(file_name))


if __name__ == '__main__':
    # main()
    # visualize_results()
    # merge_results()
    visualize_results_global()