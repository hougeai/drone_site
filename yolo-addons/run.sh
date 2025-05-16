if [ ! -d "logs" ]; then
    mkdir logs
fi

# kill $(ps aux | grep 'python train.py' | grep -v grep | awk '{print $2}')
CUDA_VISIBLE_DEVICES=2 python train.py > logs/yolov11m_e100.log 2>&1 &