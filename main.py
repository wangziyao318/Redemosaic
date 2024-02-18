import json
import skimage.io
import torch
from tqdm import tqdm
from tifffile.tifffile import TiffFileError

# import time

from redemosaic import redemosaic
from image_metrics import psnr, ssim

device = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")

bayer_patterns = ("gbrg", "grbg", "bggr", "rggb")
# bayer_patterns = ("bggr", "rggb")
# bayer_patterns = ("bggr",)

if __name__ == "__main__":
    print(f"torch use {device}")

    rgbimgs = skimage.io.imread_collection("img/*.TIF", conserve_memory=True, plugin="tifffile")
    imgpaths = tuple(str(rgbimgs)[1:-1].translate({ ord("'"): None}).split(', '))
    results = {}

    try:
        for rgbimg, imgpath in tqdm(zip(rgbimgs, imgpaths), total=len(rgbimgs)):
            # Redemosaic
            target = torch.tensor(rgbimg, dtype=torch.uint8, device=device)
            # begin = time.time()
            preds = redemosaic(target, bayer_patterns)
            # print(f"redemosaic on {device} took {time.time() - begin}s")

            # preds_test = redemosaic_test(target, bayer_patterns)

            # assert torch.sum(torch.ne(preds, preds_test)).item() == 0

            # PSNR, SSIM
            # begin = time.time()
            psnr_o = psnr(preds, target)
            # print(f"psnr on {device} took {time.time() - begin}s")
            # begin = time.time()
            ssim_o = ssim(preds, target)
            # print(f"ssim on {device} took {time.time() - begin}s")

            # Store results
            results[imgpath] = {}
            results[imgpath]["psnr"] = {}
            results[imgpath]["ssim"] = {}
            for bayer_pattern, psnr_i, ssim_i in zip(bayer_patterns, psnr_o, ssim_o):
                results[imgpath]["psnr"][bayer_pattern] = psnr_i.item()
                results[imgpath]["ssim"][bayer_pattern] = ssim_i.item()
    except TiffFileError:
        print(f"ERROR: corrupted file detected after {imgpath}")
    finally:
        with open("results.json", "w") as f:
            json.dump(results, f)