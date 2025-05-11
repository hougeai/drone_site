import re
import os
import numpy as np
import matplotlib.pyplot as plt
import os.path as osp
root_dir = './logs/'
paths = [
    f'{root_dir}/aug_e12.log',
]

def plot_map():
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    for path in paths:
        map, map5, map75, maps = [], [], [], []
        lines = open(path,'r').read().splitlines()
        for line in lines:
            if line.startswith(' Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all'):
                map.append(float(line.split(' ')[-1]))
            if line.startswith(' Average Precision  (AP) @[ IoU=0.50      |'):
                map5.append(float(line.split(' ')[-1]))
            if line.startswith(' Average Precision  (AP) @[ IoU=0.75      |'):
                map75.append(float(line.split(' ')[-1]))
            if line.startswith(' Average Precision  (AP) @[ IoU=0.50:0.95 | area= small'):
                maps.append(float(line.split(' ')[-1]))
        label = osp.basename(path).split('.')[0]
        print(label, np.argmax(map), f'{np.max(map):.3f}', f'{map5[np.argmax(map)]:.3f}', f'{map75[np.argmax(map)]:.3f}', f'{maps[np.argmax(map)]:.3f}')
        plt.plot(np.arange(0, len(map)), map, label=label)
    plt.legend(loc='best')
    plt.savefig('1.jpg')

if __name__ == '__main__':
    plot_map()