import os
import json
import torch
import asyncio
from skimage.io import imread_collection
from tqdm.asyncio import tqdm

from redemosaic import redemosaic
from image_metrics import psnr, ssim, vmaf


# torch enable gpu acceleration if possible
device = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")

# batch size 4 in cuda requires at least 13G VRAM and 17G RAM.
# batch size 2 in cuda requires at least G VRAM and G RAM.
# if use cpu, please do batch size 1 for 4 times and set update_results to True.
# bayer_patterns is a tuple, so "," is required when only one element exists.
bayer_patterns = ("gbrg", "grbg", "bggr", "rggb")
# bayer_patterns = ("bggr", "rggb")
# bayer_patterns = ("bggr",)

# do not overwrite results file
update_results = False

# enable calculating VMAF
use_vmaf = True
# enable cuda acceleration of VMAF, requires compiled libvmaf and ffmpeg with cuda enabled
# in practice, libvmaf_cuda is slower than libvmaf on cpu for single image comparison, but is faster for video (batch of frames) comparison
# it's recommended to leave it disabled
libvmaf_cuda = False
# max concurrent vmaf cuda tasks on gpu, prevent CUDA out of memory error, extra vmaf tasks are allocated to cpu
# only works when libvmaf_cuda = True
vmaf_cuda_concurrent = 4
# set tuple of VMAF versions, recommended to use "vmaf_v0.6.1" only because "vmaf_4k_v0.6.1" results are bigger and often 100.0
vmaf_versions = ("vmaf_v0.6.1", "vmaf_4k_v0.6.1")
# vmaf_versions = ("vmaf_v0.6.1",)

# under PYTHONPATH
target_folder = "img"
preds_folder = "output"
results_file = "results.json"

# input image extension
target_ext = ".TIF"

async def main():
    """
    async main() function for asyncio compatibility.
    """
    print(f"torch use {device}")

    # input images, extension is case sensitive
    targets = imread_collection(os.path.join(target_folder, "*" + target_ext), conserve_memory=True)
    # input image names with extension
    target_names = str(targets)[1:-1].translate({ord("'"): None}).replace(target_folder + "/", "").split(', ')
    B = len(targets)

    # continue with existing results if update_results is True
    results = {}
    if update_results:
        try:
            with open(results_file, "r") as f:
                results = json.load(f)
        except json.decoder.JSONDecodeError or OSError:
            print(f"{results_file} corrupted, please set update_results to False to overwrite it")
            return
    
    vmaf_cuda_tasks = []
    # main function
    try:
        # TaskGroup() is introduced in Python3.11, please upgrade Python if you encounter error
        async with asyncio.TaskGroup() as tg:
            # for each input image
            async for target, target_name in tqdm(zip(targets, target_names), total=B):
                # init tensor of input image
                target = torch.tensor(target, dtype=torch.uint8, device=device)

                # Redemosaic input image on each Bayer pattern, generate a batch of redemosaiced output images
                preds = redemosaic(target, bayer_patterns)

                # init results dict with input image name, takes account in update results case
                if target_name not in results.keys():
                    results[target_name] = {}

                # calculate VMAF
                if use_vmaf:
                    # limit number of concurrent vmaf cuda tasks, prevent CUDA out of memory and send extra tasks to cpu
                    if libvmaf_cuda and sum([not task.done() for task in vmaf_cuda_tasks]) < vmaf_cuda_concurrent:
                        vmaf_cuda_tasks.append(tg.create_task(vmaf(preds, preds_folder, target_name, target_folder, results, bayer_patterns, vmaf_versions, libvmaf_cuda)))
                    else:
                        tg.create_task(vmaf(preds, preds_folder, target_name, target_folder, results, bayer_patterns, vmaf_versions, False))
                    # calling await after create_task() is necessary for the task to actually start running
                    # eager_task_factory() introduced in Python3.12 could overcome this, but torch-cuda not provides package for Python3.12 yet
                    await asyncio.sleep(0)

                # calculate PSNR and SSIM
                psnr_o = psnr(preds, target)
                ssim_o = ssim(preds, target)

                # Store PSNR and SSIM results in dict
                if "psnr" not in results[target_name].keys():
                    results[target_name]["psnr"] = {}
                if "ssim" not in results[target_name].keys():
                    results[target_name]["ssim"] = {}
                for bayer_pattern, psnr_i, ssim_i in zip(bayer_patterns, psnr_o, ssim_o):
                    results[target_name]["psnr"][bayer_pattern] = psnr_i.item()
                    results[target_name]["ssim"][bayer_pattern] = ssim_i.item()
    except ExceptionGroup:
        print(f"Error found near {target_name}")
        raise
    finally:
        del targets, target_names
        print(f"VMAF on CUDA: {len(vmaf_cuda_tasks)}it\nVMAF on CPU: {B - len(vmaf_cuda_tasks)}it")
        with open(results_file, "w") as f:
            json.dump(results, f)
        print(f"Results written to {results_file}")

if __name__ == "__main__":
    asyncio.run(main())