<div align="center">

# Drone Site
</div>

## 0. 目录结构

```bash
.
├── README.md         # 说明文档
├── requirements.txt  # 依赖包
├── slice_image.py    # 切割图像代码
├── slice.sh          # 切割图像脚本
├── util.py           # 工具函数
├── rtdetrv2_pytorch/ # RTDETRv2 仓库代码    
├── ultralytics/      # Ultralytics-yolo 仓库代码(ignore)
├── inference/        # 两个模型的推理测试
└── yolo-addons/      # yolo 训练配置/脚本
```

## 1. 环境准备
### 1.1 下载本项目
```bash
git clone https://github.com/hougeai/drone_site.git
cd drone_site
```
### 1.2 创建 conda 环境
```bash
conda create -n cv python=3.12
conda activate cv
pip install -r requirements.txt
```
### 1.3 下载模型权重

百度网盘：

```bash
https://pan.baidu.com/s/1D_cgxulo4GfQGHN7wZSB2Q?pwd=gvr6
```

权重列表：
```bash
inference/ckpt/
├── rtdetr2_e36_best.pth
└── yolo11m_e100_best.pt
```
## 2. 模型推理

```bash
cd inference
```

### 2.1 RTDETRv2 推理

```bash
python inference_rtdetr2.py \
    -w /path/to/weights \ # 下载的模型权重路径
    -i /path/to/image \ # 测试图片路径
    -s 0.4 \ # 置信度阈值
```

### 2.2 yolov11 推理

```bash
python inference_yolo11 \
    -w /path/to/weights \ # 下载的模型权重路径
    -i /path/to/image \ # 测试图片路径
    -s 0.2 \ # 置信度阈值
```

### 2.3 推理时长

```bash
推理设备：RTX 4080 16G
图像尺寸：4k
图像切割：640x640 overlap 0.25，每张图像共切割为 40 张小图

rtdetrv2 推理时长：0.88s/整图
yolov11m 推理时长：1.67s/整图
```