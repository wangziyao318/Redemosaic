import json
import skimage.io
import torch
from tqdm import tqdm
from tifffile.tifffile import TiffFileError

from redemosaic import redemosaic
from image_metrics import psnr, ssim

device = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")

bayer_patterns = ("gbrg", "grbg", "bggr", "rggb")

if __name__ == "__main__":
    print(f"torch use {device}")

    rgbimgs = skimage.io.imread_collection("img/*.TIF")
    imgpaths = str(rgbimgs)[1:-1].split(', ')

    results = {}
    i = 0
    try:
        for rgbimg in tqdm(rgbimgs):
            target = torch.tensor(rgbimg, dtype=torch.uint8, device=device)
            preds = redemosaic(target, bayer_patterns)

            results[imgpaths[i][1:-1]] = {}
            results[imgpaths[i][1:-1]]["psnr"] = {}
            results[imgpaths[i][1:-1]]["ssim"] = {}
            
            psnr_o = psnr(preds, target)
            ssim_o = ssim(preds, target)
            
            for bayer_pattern, psnr_i, ssim_i in zip(bayer_patterns, psnr_o, ssim_o):
                results[imgpaths[i][1:-1]]["psnr"][bayer_pattern] = psnr_i.item()
                results[imgpaths[i][1:-1]]["ssim"][bayer_pattern] = ssim_i.item()

            i = i + 1
    except TiffFileError:
        print(f"ERROR: corrupted file {imgpaths[i]}")
    finally:
        with open("results.json", "w") as f:
            json.dump(results, f)