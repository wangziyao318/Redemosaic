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

    rgbimgs = skimage.io.imread_collection("imgtest/*.TIF")
    imgpaths = str(rgbimgs)[1:-1].split(', ')

    i = 0
    try:
        for rgbimg in tqdm(rgbimgs):
            target = torch.tensor(rgbimg, dtype=torch.uint8, device=device)
            preds = redemosaic(target, bayer_patterns)
            print(psnr(preds, target.unsqueeze(0).repeat(4, 1, 1, 1)))
            print(ssim(preds, target.unsqueeze(0).repeat(4, 1, 1, 1)))



            i = i + 1
    except TiffFileError:
        print(f"{imgpaths[i]} corrupted")
