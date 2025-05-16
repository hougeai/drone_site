import os
import cv2
import json
import shutil
import numpy as np
from tqdm import tqdm
from collections import defaultdict

def get_img_set_dict():
    root_dir = '/mnt/sda1/datasets/drone_site/site_photo'
    img_files = sorted([f for f in os.listdir(root_dir) if f.endswith('.jpg')])
    img_sets = set()
    for img_file in img_files:
        set_name = '_'.join(img_file.split('_')[:-2])
        img_sets.add(set_name)
    print(sorted(list(img_sets)))
    img_set_dict = defaultdict(list)
    for img_file in img_files:
        set_name = '_'.join(img_file.split('_')[:-2])
        img_set_dict[set_name].append(img_file)
    for set_name, img_files in img_set_dict.items():
        img = cv2.imread(os.path.join(root_dir, img_files[0]))
        ih, iw, ic = img.shape
        print(set_name, len(img_files), ih, iw)
    return img_set_dict

def draw_boxes():
    # img_dir = '/mnt/sda1/datasets/drone_site/dataset/images/val'
    # ann_dir = '/mnt/sda1/datasets/drone_site/dataset/labels/val'
    img_dir = '/mnt/sda1/datasets/drone_site/site_photo'
    ann_dir = '/mnt/sda1/datasets/drone_site/site_photo'
    img_file = '0001_V_12.jpg'
    ann_file = img_file.replace('.jpg', '.txt')
    img = cv2.imread(os.path.join(img_dir, img_file))
    ih, iw, ic = img.shape
    print(img_file, ih, iw, ic)
    # 读取yolo格式标注文件
    ann = np.loadtxt(os.path.join(ann_dir, ann_file), dtype=np.float32).reshape(-1, 5)
    # 画框
    for box in ann:
        cls_id, xc, yc, w, h = box
        x1, y1, x2, y2 = int((xc - w/2)*iw), int((yc - h/2)*ih), int((xc + w/2)*iw), int((yc + h/2)*ih)
        # print(cls_id, x1, y1, x2, y2)
        color = (0, 255, 0) if cls_id == 0 else (255, 0, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, str(cls_id), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.imwrite(img_file, img)

def yolo2coco():
    root_dir = '/mnt/sda1/datasets/drone_site/site_photo'
    img_files = sorted([f for f in os.listdir(root_dir) if f.endswith('.jpg')])
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 0, "name": "waste"},
        ]
    }
    annotation_id = 1
    # 处理每个图片和其标注
    for image_id, img_file in tqdm(enumerate(img_files, 1), total=len(img_files)):
        # 读取图片获取尺寸
        img = cv2.imread(os.path.join(root_dir, img_file))
        ih, iw, _ = img.shape
        # 添加图片信息
        coco_format["images"].append({
            "id": image_id,
            "width": iw,
            "height": ih,
            "file_name": img_file
        })
        # 读取对应的标注文件
        ann_file = img_file.replace('.jpg', '.txt')
        if not os.path.exists(os.path.join(root_dir, ann_file)):
            continue
        try:
            ann = np.loadtxt(os.path.join(root_dir, ann_file), dtype=np.float32).reshape(-1, 5)
            # 处理每个标注框
            for box in ann:
                cls_id, xc, yc, w, h = box
                if cls_id > 0: # 只取用建筑垃圾类别
                    continue
                # YOLO格式转COCO格式（归一化坐标转像素坐标）
                width_pixel = w * iw
                height_pixel = h * ih
                x_pixel = (xc - w/2) * iw
                y_pixel = (yc - h/2) * ih
                # 创建COCO格式的标注
                coco_ann = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": int(cls_id),
                    "bbox": [int(x_pixel), int(y_pixel), int(width_pixel), int(height_pixel)],
                    "area": int(width_pixel * height_pixel),
                    "iscrowd": 0
                }
                coco_format["annotations"].append(coco_ann)
                annotation_id += 1
        except Exception as e:
            print(f"Error processing {ann_file}: {str(e)}")
            continue
    # 保存为COCO格式的JSON文件
    output_file = os.path.join(root_dir, '../dataset/annotations/trainval.json')
    with open(output_file, 'w') as f:
        json.dump(coco_format, f, indent=2)
    print(f"转换完成，共处理{len(coco_format['images'])}张图片，{len(coco_format['annotations'])}个标注")

def split_trainval():
    root_dir = '/mnt/sda1/datasets/drone_site'
    img_set_dict = get_img_set_dict()
    train, val = [], []
    # 首先移动图片和标注
    for set_name, img_files in img_set_dict.items():
        img_files = sorted(img_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        train_len = int(len(img_files) * 0.85)
        print(set_name, len(img_files), train_len, img_files[0], img_files[-1])
        train.extend(img_files[:train_len])
        val.extend(img_files[train_len:])
    print(f"训练集：{len(train)}张，验证集：{len(val)}张")
    # os.makedirs(os.path.join(root_dir, 'dataset/images/train'), exist_ok=True)
    # os.makedirs(os.path.join(root_dir, 'dataset/images/val'), exist_ok=True)
    # os.makedirs(os.path.join(root_dir, 'dataset/labels/train'), exist_ok=True)
    # os.makedirs(os.path.join(root_dir, 'dataset/labels/val'), exist_ok=True)
    # for img_file in train:
    #     src_file = os.path.join(root_dir, 'site_photo', img_file)
    #     dst_file = os.path.join(root_dir, 'dataset/images/train', img_file)
    #     shutil.copy(src_file, dst_file)
    #     src_file = os.path.join(root_dir, 'site_photo', img_file.replace('.jpg', '.txt'))
    #     dst_file = os.path.join(root_dir, 'dataset/labels/train', img_file.replace('.jpg', '.txt'))
    #     shutil.copy(src_file, dst_file)
    # for img_file in val:
    #     src_file = os.path.join(root_dir, 'site_photo', img_file)
    #     dst_file = os.path.join(root_dir, 'dataset/images/val', img_file)
    #     shutil.copy(src_file, dst_file)
    #     src_file = os.path.join(root_dir, 'site_photo', img_file.replace('.jpg', '.txt'))
    #     dst_file = os.path.join(root_dir, 'dataset/labels/val', img_file.replace('.jpg', '.txt'))
    #     shutil.copy(src_file, dst_file)
    # print("复制完成")
    # 生成 coco 格式的标注 train val
    trainval_coco = json.load(open(os.path.join(root_dir, 'dataset/annotations/trainval.json')))
     # 创建训练集和验证集的COCO格式数据
    train_coco = {
        "images": [],
        "annotations": [],
        "categories": trainval_coco["categories"]
    }
    val_coco = {
        "images": [],
        "annotations": [],
        "categories": trainval_coco["categories"]
    }
    
    # 建立训练集和验证集的图片名称集合，用于快速查找
    train_set = set(train)
    val_set = set(val)
    # 划分图片
    for img_info in trainval_coco["images"]:
        if img_info["file_name"] in train_set:
            train_coco["images"].append(img_info)
        elif img_info["file_name"] in val_set:
            val_coco["images"].append(img_info)
    # 建立图片ID到数据集的映射
    train_img_ids = {img["id"] for img in train_coco["images"]}
    val_img_ids = {img["id"] for img in val_coco["images"]}
    # 划分标注
    for ann in trainval_coco["annotations"]:
        if ann["image_id"] in train_img_ids:
            train_coco["annotations"].append(ann)
        elif ann["image_id"] in val_img_ids:
            val_coco["annotations"].append(ann)
    # 保存为COCO格式的JSON文件
    train_file = os.path.join(root_dir, 'dataset/annotations/train.json')
    val_file = os.path.join(root_dir, 'dataset/annotations/val.json')
    with open(train_file, 'w') as f:
        json.dump(train_coco, f, indent=2)
    with open(val_file, 'w') as f:
        json.dump(val_coco, f, indent=2)
    print(f"训练集：{len(train_coco['images'])}张图片，{len(train_coco['annotations'])}个标注")
    print(f"验证集：{len(val_coco['images'])}张图片，{len(val_coco['annotations'])}个标注")

def read_coco_ann():
    anno_path = '/mnt/sda1/datasets/drone_site/dataset/annotations/train_640_025.json'
    anno = json.load(open(anno_path))
    print(anno.keys())
    print(len(anno['images']), len(anno['annotations']))

def coco2yolo():
    root_dir = '/mnt/sda1/datasets/drone_site'
    anno = json.load(open(os.path.join(root_dir, 'dataset/annotations/train_640_025.json')))
    label_dir = os.path.join(root_dir, 'dataset/labels/train_640_025')
    os.makedirs(label_dir, exist_ok=True)
    # 建立图片ID到标注的映射
    img_ann_dict = defaultdict(list)
    for ann in anno['annotations']:
        img_ann_dict[ann['image_id']].append(ann)
    # 处理每张图片
    for img_info in tqdm(anno['images']):
        img_file = img_info['file_name']
        img_id = img_info['id']
        img_w = img_info['width']
        img_h = img_info['height']
        # 获取该图片的所有标注
        annotations = img_ann_dict[img_id]
        # 如果没有标注，创建空文件
        label_file = os.path.join(label_dir, img_file.replace('.jpg', '.txt'))
        if not annotations:
            open(label_file, 'w').close()
            continue
        # 转换标注格式
        yolo_anns = []
        for ann in annotations:
            # COCO bbox: [x, y, width, height]
            x, y, w, h = ann['bbox']
            # 转换为YOLO格式：归一化的中心点坐标和宽高
            x_center = (x + w/2) / img_w
            y_center = (y + h/2) / img_h
            width = w / img_w
            height = h / img_h
            category_id = ann['category_id']
            # YOLO格式：class_id x_center y_center width height
            yolo_anns.append(f"{category_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        # 保存YOLO格式标注
        with open(label_file, 'w') as f:
            f.write('\n'.join(yolo_anns))

if __name__ == '__main__':
    # draw_boxes()
    # yolo2coco()
    # split_trainval()
    # read_coco_ann()
    coco2yolo()
