import torchmetrics.functional.image
import torch

import skimage.metrics

import image_metrics

import cv2 as cv

import time

device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")

preds = torch.tensor(cv.imread("imgtest/r0cea5432t.TIF")[:,:,[2,1,0]], dtype=torch.float32, device=device)
targets = torch.tensor(cv.imread("imgtest/r0d0ff4c2t.TIF")[:,:,[2,1,0]], dtype=torch.float32, device=device)


# print(f"torchmetrics {torchmetrics.functional.image.peak_signal_noise_ratio(preds, targets, data_range=255)}")

print(f"skimage {skimage.metrics.peak_signal_noise_ratio(targets.cpu().detach().numpy(), preds.cpu().detach().numpy(), data_range=255)}")

print(f"ours {image_metrics.psnr(torch.stack((preds, preds, preds)), torch.stack((targets, targets, targets)))}")

print("\n=======================\n")


rgb2y = torch.tensor([0.2989, 0.5870, 0.1140], dtype=torch.float32, device=device)

p_y = (preds.float() @ rgb2y)
t_y = (targets.float() @ rgb2y)


# begin = time.time()
# print(f"torchmetrics {torchmetrics.functional.image.structural_similarity_index_measure(p_y[None, None, ...], t_y[None, None, ...], gaussian_kernel=False, data_range=255.)} in {time.time() - begin}s")

begin = time.time()
print(f"skimage {skimage.metrics.structural_similarity(p_y.cpu().detach().numpy(), t_y.cpu().detach().numpy(), data_range=255.)} in {time.time() - begin}s")

begin = time.time()
print(f"ours {image_metrics.ssim(torch.stack((preds, preds, preds, preds)), torch.stack((targets, targets, targets, targets)))} in {time.time() - begin}s")