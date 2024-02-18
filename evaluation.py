import torchmetrics.functional.image
import torch

import skimage.metrics

import metricsimpl

import cv2 as cv

# device = torch.device("cuda" if torch.cuda.is_available()
#                           else "mps" if torch.backends.mps.is_available()
#                           else "cpu")

preds = torch.tensor(cv.imread("out.TIF")[:,:,[2,1,0]], dtype=torch.float32)
target = torch.tensor(cv.imread("in.TIF")[:,:,[2,1,0]], dtype=torch.float32)


print(f"torchmetrics {torchmetrics.functional.image.peak_signal_noise_ratio(preds, target, data_range=255)}")

print(f"skimage {skimage.metrics.peak_signal_noise_ratio(target.numpy(), preds.numpy(), data_range=255)}")

print(f"ours {metricsimpl.PSNR(preds, target)}")



print("\n=======================\n")


rgb2y = torch.tensor([0.2989, 0.5870, 0.1140], dtype=torch.float32)

p_y = (preds @ rgb2y).float()
t_y = (target @ rgb2y).float()


# bgt = time.time()
# print("\ntorchmetrics")
# print(f"{torchmetrics.functional.image.structural_similarity_index_measure(xt_y, yt_y, gaussian_kernel=False, data_range=255.)}")

# print(f"ssimtorch {time.time() - bgt}s")


# xn = x.numpy()
# yn = y.numpy()

# bgs = time.time()
print(f"skimage {skimage.metrics.structural_similarity(p_y.numpy(), t_y.numpy(), data_range=255.)}")

# print(f"ssimskit {time.time() - bgs}s")

# print("\n===================\n")
print(f"torch {metricsimpl.SSIM(preds[None, ...], target[None, ...])}")