import cv2
import os
import argparse
import glob
import numpy as np
from utils import *
from networks import *
from SSIM import ssim
from pathlib import Path
import time 
import math
import sys

parser = argparse.ArgumentParser(description="PReNet_Test")
parser.add_argument("--logdir", type=str, default="logs/PReNet6/", help='path to model and log files')
parser.add_argument("--data_path", type=str, default="../datasets/test/Rain100H/rainy", help='path to training data')
parser.add_argument("--save_path", type=str, default="./results/PReNet", help='path to save results')
parser.add_argument("--recurrent_iter", type=int, default=6, help='number of recursive stages')
opt = parser.parse_args()


# input is uint8 with Y channel of YCbCr
def matlab_psnr(img1, img2):
    diff = img1 - img2
    mse = np.mean(np.square(diff))
    PSNR = 10*math.log10(255. * 255. / mse)
    return PSNR

def main():
    os.makedirs(opt.save_path, exist_ok=True)

    # Build model
    print('Loading model ...\n')
    model = PReNet_r(opt.recurrent_iter)
    print_network(model)
    model.set_state_dict(paddle.load(os.path.join(opt.logdir, 'net_latest111.pdparams')))
    model.eval()
    time_test = 0
    count = 0
    ssim_all = []
    psnr_all = []
    for img_name in os.listdir(opt.data_path):
        if is_image(img_name):
            img_path = os.path.join(opt.data_path, img_name)
            # input image
            y = cv2.imread(img_path)
            b, g, r = cv2.split(y)
            y = cv2.merge([r, g, b])

            y = normalize(np.float32(y))
            y = np.expand_dims(y.transpose(2, 0, 1), 0)
            y = paddle.to_tensor(y)

            with paddle.no_grad(): #
                paddle.device.cuda.synchronize()
                start_time = time.time()

                out, _ = model(y)
                
                out = paddle.clip(out, 0., 1.)
                paddle.device.cuda.synchronize()
                end_time = time.time()
                dur_time = end_time - start_time
                time_test += dur_time

                print(img_name, ': ', dur_time)

            save_out = np.uint8(255 * out.cpu().numpy().squeeze())   #back to cpu
            save_out = save_out.transpose(1, 2, 0)
            b, g, r = cv2.split(save_out)
            save_out = cv2.merge([r, g, b])

            cv2.imwrite(os.path.join(opt.save_path, img_name), save_out)

            count += 1

    print('Avg. time:', time_test/count)


if __name__ == "__main__":
    main()

