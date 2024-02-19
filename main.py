import asyncio
import json
from skimage.io import imread_collection
import glob
import ntpath
import torch
import os
from tqdm.asyncio import tqdm
from tifffile.tifffile import TiffFileError

import time

from redemosaic import redemosaic
from image_metrics import psnr, ssim, vmaf

device = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")

bayer_patterns = ("gbrg", "grbg", "bggr", "rggb")
# bayer_patterns = ("bggr", "rggb")
# bayer_patterns = ("bggr",)

update_results = False
# use_vmaf = True

# under PYTHONPATH=. by default
targets_folder = "img"
preds_folder = "output"
results_file = "results.json"

async def main():
    print(f"torch use {device}")

    rgbimgs = imread_collection(os.path.join(targets_folder, "*.TIF"), conserve_memory=True, plugin="tifffile")
    imgnames = tuple([ntpath.basename(i) for i in glob.glob(os.path.join(targets_folder, "*.TIF"))])

    results = {}
    if update_results:
        try:
            with open(results_file, "r") as f:
                results = json.load(f)
        except json.decoder.JSONDecodeError:
            print("results file corrupted, please set update_results to True")
    
    try:
        async with asyncio.TaskGroup() as tg:
            async for rgbimg, imgname in tqdm(zip(rgbimgs, imgnames), total=len(imgnames)):

                # Redemosaic
                target = torch.tensor(rgbimg, dtype=torch.uint8, device=device)
                # begin = time.time()
                preds = redemosaic(target, bayer_patterns)
                # print(f"redemosaic on {device} took {time.time() - begin}s")

                # init results dict for this image
                results[imgname] = {}

                # async calculation of vmaf
                tg.create_task(vmaf(preds, imgname, targets_folder, preds_folder, bayer_patterns, results))
                # await after create_task() is necessary for the task to start running
                await asyncio.sleep(0)

                # PSNR, SSIM
                # begin = time.time()
                psnr_o = psnr(preds, target)
                # print(f"psnr on {device} took {time.time() - begin}s")
                # begin = time.time()
                ssim_o = ssim(preds, target)
                # print(f"ssim on {device} took {time.time() - begin}s")

                # Store results for PSNR and SSIM
                results[imgname]["psnr"] = {}
                results[imgname]["ssim"] = {}
                for bayer_pattern, psnr_i, ssim_i in zip(bayer_patterns, psnr_o, ssim_o):
                    results[imgname]["psnr"][bayer_pattern] = psnr_i.item()
                    results[imgname]["ssim"][bayer_pattern] = ssim_i.item()

    except TiffFileError:
        print(f"ERROR: corrupted file detected after {imgname}")
    except asyncio.CancelledError:
        print("async task cancelled")
    finally:
        with open(results_file, "w") as f:
            json.dump(results, f)

if __name__ == "__main__":
    mst = time.time()
    asyncio.run(main())
    print(f"overall time {time.time() - mst}s")