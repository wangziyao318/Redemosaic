import torch
import skimage.metrics
import image_metrics
import cv2 as cv
import time

device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")

preds = torch.tensor(cv.imread("imgtest/r0cea5432t.TIF")[:,:,[2,1,0]], dtype=torch.float32, device=device)
target = torch.tensor(cv.imread("imgtest/r0d0ff4c2t.TIF")[:,:,[2,1,0]], dtype=torch.float32, device=device)

begin = time.time()
print(f"skimage psnr {skimage.metrics.peak_signal_noise_ratio(target.cpu().detach().numpy(), preds.cpu().detach().numpy(), data_range=255)} in {time.time() - begin}s")

begin = time.time()
print(f"ours psnr {image_metrics.psnr(preds[None, ...], target).item()} in {time.time() - begin}s")

print("\n=======================\n")

rgb2y = torch.tensor([0.2989, 0.5870, 0.1140], dtype=torch.float32, device=device)

p_y = (preds.float() @ rgb2y)
t_y = (target.float() @ rgb2y)

begin = time.time()
print(f"skimage ssim {skimage.metrics.structural_similarity(p_y.cpu().detach().numpy(), t_y.cpu().detach().numpy(), data_range=255.)} in {time.time() - begin}s")

begin = time.time()
print(f"ours ssim {image_metrics.ssim(preds[None, ...], target).item()} in {time.time() - begin}s")