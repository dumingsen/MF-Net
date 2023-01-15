import os
import sys
dir_mytest = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, dir_mytest)
import torch
from dataloader import read_UEA
from UtilCollection.util import random_seed
from torch.autograd import Variable
from UtilCollection.tsne import gather_by_tsne
from UtilCollection.tsne import gather_all_by_tsne
import seaborn as sns
import numpy as np
from dataloader.read_UEA import load_UEA
import argparse
from draw_picture.grad_cam import GradCAM
import matplotlib.pyplot as plt
sns.set()
print('当前使用的pytorch版本：', torch.__version__)

parser = argparse.ArgumentParser(description='WiWo transformer for TSC')

parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
# parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--length', type=int, default=8192, help='Embedding length')
parser.add_argument('--writer_path', type=str, default='runs/exp', help='TensorBoard path')
parser.add_argument('--data_path', type=str, default='G:/桌面/MF-Net-1128 - 副本/data')
parser.add_argument('--seed', type=int, default=40, help='random seed')
parser.add_argument('--dropout', type=float, default=0.05, help='attention dropout rate')
parser.add_argument('--n_epochs', type=int, default=100)
parser.add_argument('--cache_path', type=str, default='G:/桌面/MF-Net-1128 - 副本/cache')
parser.add_argument('--model_path', type=str,
                    default=r'G:/桌面/MF-Net-1128 - 副本/saved_model/AtrialFibrillation/AtrialFibrillation batch=16 length=3072 window=64.pkl')
#python draw_picture/cam_show.py
args = parser.parse_args()

random_seed(args.seed)

file_name = args.model_path.split('/')[-1].split(' ')[0]

# path = f'./data/MTS_dataset/ECG/ECG.mat'

heatMap_or_not = False  # 是否绘制Score矩阵的HeatMap图
gather_or_not = True  # 是否绘制单个样本的step和channel上的聚类图
gather_all_or_not = True  # 是否绘制所有样本在特征提取后的聚类图

net = torch.load(args.model_path, map_location=torch.device(args.device))
# net = torch.load(args.model_path)

for (name, m) in net.named_modules():
    print(name)
    #module.EncoderList.2.layers.0.1.mlp_block.fn.fn.net.2
    if name == 'attention.0.EncoderList.0.layers.0.0.mlp_block.fn.fn.net.2':#
        target_layers = [m]
        print(1)

cam = GradCAM(model=net, target_layers=target_layers, use_cuda=False)
target_category = 7

correct = 0
total = 0
train_loader, test_loader, num_class = load_UEA(file_name, args)
#train_loader= torch.unsqueeze(train_loader, dim=1)

all_sample_X = []
all_sample_Y = []


def show_cam_on_image(ts, cam):
    pass


for i,(x, y) in enumerate(test_loader):
    x = torch.unsqueeze(x, dim=1)
    #y = torch.unsqueeze(y, dim=1)
    x = Variable(x).float().to(args.device)
    y = Variable(y).to(args.device)
    target_category = y
    print('x',x.shape)#[1, 1, 2, 640]
    print('y',y.shape)#[1]
    grayscale_cam = cam(input_tensor=x, target_category=target_category)#

    # embedding, encoder, output, pred_y = net(x)
    grayscale_cam = grayscale_cam[0, :]
    c = np.exp(grayscale_cam) / np.sum(np.exp(grayscale_cam))
    plt.plot(x[0][0][0].cpu().detach().numpy())
    # print(np.arange(640).shape)
    # print('x[0][0]',x[0][0].shape)
    plt.scatter(np.arange(640), x[0][0][0].cpu().detach().numpy(), c=c)#144
   #plt.show()
    # visualization = show_cam_on_image(x,cam)
    # plt.imshow(visualization)
    # plt.colorbar()

    if os.path.exists(f'CAM') == False:
        os.makedirs(f'CAM')
    plt.savefig('CAM_figure{0}{1}.pdf'.format(i)(file_name), format='pdf')

    plt.show()
