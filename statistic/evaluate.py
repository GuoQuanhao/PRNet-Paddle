from PIL import Image
import numpy as np
from tqdm import tqdm
import argparse
import paddle.nn.functional as F
import paddle
import math


parser = argparse.ArgumentParser(description="PReNet Evaluate")
parser.add_argument("--gt_path", type=str, default='../../datasets/test/Rain100H/', help='path to ground truth')
parser.add_argument("--pred_path", type=str, default='../results/Rain100H/PReNet/', help='path to pred')
opt = parser.parse_args()


mat = np.array(
    [[ 65.481, 128.553, 24.966 ],
     [-37.797, -74.203, 112.0  ],
     [  112.0, -93.786, -18.214]])

mat_inv = np.linalg.inv(mat)
offset = np.array([16, 128, 128])


# Convert RGB to YCbCr
def rgb2ycbcr(rgb_img):
    ycbcr_img = np.zeros(rgb_img.shape)
    for x in range(rgb_img.shape[0]):
        for y in range(rgb_img.shape[1]):
            ycbcr_img[x, y, :] = np.dot(mat, rgb_img[x, y, :]) + offset
    return ycbcr_img / 255.0


# Gaussian lowpass filter
def gauss_filter(kernel_size=11, sigma=1.5):
    max_idx = kernel_size // 2
    idx = np.linspace(-max_idx, max_idx, kernel_size)
    Y, X = np.meshgrid(idx, idx)
    gauss_filter = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    gauss_filter /= np.sum(np.sum(gauss_filter))
    return gauss_filter


# matlab psnr
def matlab_psnr(img1, img2):
    diff = img1 - img2
    mse = np.mean(np.square(diff))
    PSNR = 10 * math.log10(1.0 / mse)
    return PSNR


# matlab ssim
def matlab_ssim(img1, img2, window=None, C1=6.5, C2=58.5):
    mu1 = F.conv2d(img1, window)
    mu2 = F.conv2d(img2, window)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return paddle.mean(ssim_map).item()


def main():
    psnrs = 0
    ssims = 0
    gt_path=opt.gt_path
    pred_path = opt.pred_path
    window = paddle.to_tensor(gauss_filter(kernel_size=11)).unsqueeze(0).unsqueeze(0)
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    for i in tqdm(range(1, 101)):
        x_true = rgb2ycbcr(np.array(Image.open(gt_path + 'norain-{:03d}.png'.format(i)), dtype=np.float64) / 255.0)[:,:,0]
        x = rgb2ycbcr(np.array(Image.open(pred_path + 'rain-{:03d}.png'.format(i)), dtype=np.float64) / 255.0)[:,:,0]
        psnrs += matlab_psnr(x, x_true)
        x = paddle.to_tensor(x).unsqueeze(0).unsqueeze(0)
        x_true = paddle.to_tensor(x_true).unsqueeze(0).unsqueeze(0)
        ssims += matlab_ssim(x * 255.0, x_true * 255.0, window, C1, C2)
    print('SSIM:', ssims / i, 'PSNR:', psnrs / i)

if __name__ == '__main__':
    main()
