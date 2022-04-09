import math
import re
import numpy as np
import paddle
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import  os
import glob 


def make_grid(images, nrow, pixel=0.0):
    nrow = 8
    rows = math.ceil(images.shape[0] / 8)
    partial = rows * 8 - images.shape[0]
    imageShow = []
    for i in range(rows):
        if i == rows-1:
            imageShow.append(paddle.concat(images[nrow*i:, ...].split(nrow-partial, 0) +
            [paddle.ones([1] + images.shape[1:])*pixel] * partial, 3).squeeze(0).transpose([1,2,0]).cpu().numpy())
        else:
            imageShow.append(paddle.concat(images[nrow*i:nrow*(i+1), ...].split(8, 0), 3).squeeze(0).transpose([1,2,0]).cpu().numpy())
    return np.concatenate(imageShow, 0)


def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir, '*epoch*.pdparams'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*epoch(.*).pdparams.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch


def batch_PSNR(img, imclean, data_range):
    Img = img.cpu().numpy().astype(np.float32)
    Iclean = imclean.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
    return (PSNR/Img.shape[0])


def normalize(data):
    return data / 255.


def is_image(img_name):
    if img_name.endswith(".jpg") or img_name.endswith(".bmp") or img_name.endswith(".png"):
        return True
    else:
        return  False


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


