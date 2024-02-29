import os
import torch
import time
from pathlib import Path
from skimage.io import imread_collection

from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from redemosaic import redemosaic
from image_metrics import peak_signal_noise_ratio as psnrme
from image_metrics import structural_similarity as ssimme

device = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")

target_dir = "fake_img"
target_ext = "png"
bayer_patterns = ["rggb", "bggr"]

targets = imread_collection(os.path.join(target_dir, "*." + target_ext), conserve_memory=True)
target_paths = str(targets)[1:-1].translate({ord("'"): None}).split(", ")
target_names = map(lambda e : Path(e).stem, target_paths)


"""
Input image: target(H,W,3).
"""
target = torch.tensor(targets[0], dtype=torch.uint8, device=device)
"""
Redemosaiced image: preds(B,H,W,3).
"""
preds = redemosaic(target, bayer_patterns)


print(psnrme(preds, target)[0].item())
bg = time.time()
print(ssimme(preds, target)[0].item())
print(f"ssim takes {time.time() - bg}s")

print("-------------------")

print(peak_signal_noise_ratio(target.cpu().detach().numpy(), preds[0].cpu().detach().numpy()))
bg = time.time()
print(structural_similarity(target.cpu().detach().numpy(), preds[0].cpu().detach().numpy(), channel_axis=2))
print(f"ssim takes {time.time() - bg}s")