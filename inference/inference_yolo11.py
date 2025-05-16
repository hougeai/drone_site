import os 
import time
import torch
import argparse
import numpy as np
from PIL import Image
from ultralytics import YOLO
from utils import multiclass_nms, draw_bbox, slice_image_


def single_image_inference(model, img, conf):
    # img = '/mnt/sda1/datasets/drone_site/dataset/images/val_640_025/DJI_20250120104640_0002_V_67_104_960_960_1600_1600.jpg'
    results = model.predict(img, conf=conf)
    for result in results:
        boxes = result.boxes
        conf = boxes.conf.cpu().numpy()
        xyxy = boxes.xyxy.cpu().numpy()
        print(f'Confidence: {conf}, Boxes: {xyxy}')

def main(args, ):
    im_file = args.image
    if not os.path.exists(im_file):
        raise FileNotFoundError(f'{im_file} not found')
    score_thr = args.score_thr
    model = YOLO(args.weight)
    # 1. slice image
    si_st = time.perf_counter()
    images_array, starting_pixels = slice_image_(im_file)
    print(f'Slicing time: {time.perf_counter() - si_st:.3f}s')
    res_boxes, res_scores = [], []
    # 2. model inference
    infer_st = time.perf_counter()
    results = model.predict(images_array, conf=0.1)
    for result, start in zip(results, starting_pixels):
        boxes = result.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        score = boxes.conf.cpu().numpy()
        box = xyxy[score > score_thr]
        score = score[score > score_thr]
        box[:, 0::2] += start[0]
        box[:, 1::2] += start[1]
        res_boxes.append(box)
        res_scores.append(score)
    print(f'model inference time: {time.perf_counter() - infer_st:.3f}s')
    # 3. NMS
    res_boxes = np.concatenate(res_boxes, axis=0)
    res_scores = np.concatenate(res_scores, axis=0)
    bboxes = torch.tensor(res_boxes, dtype=torch.float32)
    scores = torch.tensor(res_scores).unsqueeze(1)
    padding = scores.new_zeros(scores.shape[0], 1)
    scores = torch.cat([scores, padding], dim=1)
    nms_st = time.perf_counter()
    box, _ = multiclass_nms(
        bboxes,
        scores,
        score_thr=score_thr,
        nms_cfg=dict(type="nms", iou_threshold=0.1),
        max_num=600,
    )
    print(f'NMS time: {time.perf_counter() - nms_st:.3f}s, number of boxes: {len(box)}')
    box = box.numpy()
    img = Image.open(im_file).convert('RGB')
    img = draw_bbox(img, box[:, :4], box[:, 4], 'det')
    img.save(os.path.basename(im_file))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weight', type=str, default='ckpt/yolo11m_e100_best.pt'),
    parser.add_argument('-i', '--image', type=str, default='data/0001_V_12.jpg'),
    parser.add_argument('-s', '--score_thr', type=float, default=0.2),
    args = parser.parse_args()
    main(args)
