import os 
import time
import argparse
import torch 
import torch.nn as nn 
import torchvision.transforms as T
import numpy as np
from PIL import Image, ImageDraw
from sahi.slicing import slice_image
from src.core import YAMLConfig
from test import multiclass_nms, draw_bbox

class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = cfg.model.deploy()
        self.postprocessor = cfg.postprocessor.deploy()
        self.transforms = T.Compose([
                T.Resize((640, 640)),
                T.ToTensor(),
            ])
    def forward(self, images, orig_target_sizes):
        outputs = self.model(images)
        outputs = self.postprocessor(outputs, orig_target_sizes)
        return outputs
        
def main(args, ):
    device = args.device
    im_file = args.image
    if not os.path.exists(im_file):
        raise FileNotFoundError(f'{im_file} not found')
    score_thr = args.score_thr
    cfg = YAMLConfig(args.config)
    checkpoint = torch.load(args.weight, map_location='cpu')
    cfg.model.load_state_dict(checkpoint['ema']['module'])
    model = Model(cfg).to(device)
    # 1. slice image
    si_st = time.perf_counter()
    sliced_images = slice_image(im_file, slice_height=640, slice_width=640, overlap_height_ratio=0.25, overlap_width_ratio=0.25)
    print(f'Slicing time: {time.perf_counter() - si_st:.3f}s')
    images_array = sliced_images.images
    starting_pixels = sliced_images.starting_pixels
    all_data = []
    for img_array in images_array:
        pil_image = Image.fromarray(img_array).convert('RGB')  # Ensure RGB format
        im_data = model.transforms(pil_image).to(device)  # Apply transforms
        all_data.append(im_data)
    res_boxes, res_scores = [], []
    batch_tensor = torch.stack(all_data, dim=0)
    orig_target_sizes = torch.stack([torch.tensor([640, 640])] * len(all_data), dim=0).to(device) 
    # 2. model inference
    infer_st = time.perf_counter()
    with torch.no_grad():
        output = model(batch_tensor, orig_target_sizes)
        labels, boxes, scores = output
        for box, score, start in zip(boxes, scores, starting_pixels):
            box = box[score > score_thr].cpu().numpy()
            score = score[score > score_thr].cpu().numpy()
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
    parser.add_argument('-c', '--config', type=str, default='configs/rtdetrv2/dronesite.yml'),
    parser.add_argument('-w', '--weight', type=str, default='output/aug_e36/best.pth'),
    parser.add_argument('-i', '--image', type=str, default='data/0001_V_12.jpg'),
    parser.add_argument('-s', '--score_thr', type=float, default=0.4),
    parser.add_argument('-d', '--device', type=str, default='cuda')
    args = parser.parse_args()
    main(args)
