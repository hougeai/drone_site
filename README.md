<div align="center">

# Drone Site
</div>

## 使用步骤

### 1. git 本项目
```bash
git clone https://github.com/hougeai/drone_site.git
cd drone_site
```
### 2. 创建 conda 环境
```bash
conda create -n cv python=3.12
conda activate cv
pip install -r requirements.txt
```

### 3. 进入 rtdetr 目录

```bash
cd rtdetrv2_pytorch
```

### 4. 下载模型权重

百度网盘：

```bash
通过网盘分享的文件：
https://pan.baidu.com/s/1D_cgxulo4GfQGHN7wZSB2Q?pwd=gvr6
```

### 5. 单图推理测试

```bash
python inference.py \
    -w /path/to/weights \ # 下载的模型权重路径
    -i /path/to/image \ # 测试图片路径
    -s 0.4 \ # 置信度阈值
```