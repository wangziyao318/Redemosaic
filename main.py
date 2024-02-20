import os
import json
import torch
import asyncio
from pathlib import Path
from tqdm.asyncio import tqdm
from skimage.io import imread_collection

from redemosaic import redemosaic
from image_metrics import psnr, ssim, vmaf


# torch enable gpu acceleration if possible
device = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")

# batch_size = len(bayer_patterns)
# CUDA with batch size 4 requires at least 16G VRAM GPU.
# CUDA with batch size 2 requires at least 12G VRAM GPU.
# CPU with batch size 1 requires at least 16GB RAM. please do it 4 times and set update_results to True.
bayer_patterns = ["gbrg", "grbg", "bggr", "rggb"]
# bayer_patterns = ["bggr", "rggb"]
# bayer_patterns = ["bggr"]

# set to True to avoid overwritting the results file
update_results = False

# enable calculating VMAF
enable_vmaf = True
# enable cuda acceleration of VMAF, requires compiled libvmaf and ffmpeg with cuda enabled
libvmaf_cuda = True
# max concurrent vmaf cuda tasks on GPU, extra vmaf tasks are allocated to CPU to balance the load
# only works when libvmaf_cuda = True
vmaf_cuda_window_size = 128
# set VMAF versions to use
vmaf_versions = ["vmaf_v0.6.1", "vmaf_4k_v0.6.1"]
# vmaf_versions = ["vmaf_v0.6.1"]

# input image extension without dot, case sensitive
target_ext = "png"
# directory under PYTHONPATH
target_dir = "imgfake"
preds_dir = "output"
results_file = "results.json"

# specify index of input images to start, default 0. can be used as checkpoints
start_target_index = 0
# specify number of input images to process, set arbitrary big number to process all images
N = 65535

async def main():
    """
    async main() function for asyncio compatibility.
    """
    print(f"torch use {device}")

    # read input images, note that imread_collection() behaves differently from glob()
    targets = imread_collection(os.path.join(target_dir, "*." + target_ext), conserve_memory=True)
    target_paths = str(targets)[1:-1].translate({ord("'"): None}).split(", ")
    
    # set number of input images to process
    batch_count = min(N, len(target_paths) - start_target_index)

    # input filenames without ext
    target_names = map(lambda e : Path(e).stem, target_paths)

    # continue with existing results if update_results is True
    results = {}
    if update_results:
        try:
            with open(results_file, "r") as f:
                results = json.load(f)
        except OSError:
            raise
        except json.decoder.JSONDecodeError:
            print(f"\n{results_file} corrupted, please set update_results to False to overwrite it")
            return
    
    vmaf_cuda_tasks = []
    # main function
    try:
        # TaskGroup() is introduced in Python3.11, please upgrade Python if you encounter error
        async with asyncio.TaskGroup() as tg:
            # create and customize tqdm iterators
            main_iterator = tqdm(zip(targets, target_names), total=batch_count, position=0)
            main_iterator.set_description("Main")
            vmaf_iterator = tqdm(total=batch_count, position=1)
            vmaf_iterator.set_description("VMAF")

            async for target, target_name in main_iterator:
                # init tensor (H,W,3) of input image
                target = torch.tensor(target, dtype=torch.uint8, device=device)
                target_filename = target_name + "." + target_ext

                # Redemosaic input image on each Bayer pattern, generate a batch of redemosaiced output images stacked to new dimension 0
                # input: (H,W,3); output: (B,H,W,3) for B Bayer patterns
                preds = redemosaic(target, bayer_patterns)

                # init results dict with input image name, takes account in update results case
                if target_filename not in results.keys():
                    results[target_filename] = {}

                # calculate VMAF asynchronously to mitigate lag and overhead in calling external programs
                if enable_vmaf:
                    # limit number of concurrent vmaf cuda tasks, send extra tasks to cpu to prevent 'CUDA out of memory'
                    if libvmaf_cuda and sum(map(lambda task : not task.done(), vmaf_cuda_tasks)) < vmaf_cuda_window_size:
                        vmaf_cuda_tasks.append(tg.create_task(vmaf(preds, preds_dir, target_name, target_ext, target_dir, results, bayer_patterns, vmaf_versions, libvmaf_cuda, vmaf_iterator)))
                    # vmaf on cpu
                    else:
                        tg.create_task(vmaf(preds, preds_dir, target_name, target_ext, target_dir, results, bayer_patterns, vmaf_versions, False, vmaf_iterator))
                    # calling await after create_task() is necessary for the task to actually start running
                    # eager_task_factory() introduced in Python3.12 could overcome this, but torch-cuda not provides package for Python3.12 yet
                    await asyncio.sleep(0)

                # calculate PSNR and SSIM
                psnr_o = psnr(preds, target)
                ssim_o = ssim(preds, target)

                # Store PSNR and SSIM results in dict
                if "psnr" not in results[target_filename].keys():
                    results[target_filename]["psnr"] = {}
                if "ssim" not in results[target_filename].keys():
                    results[target_filename]["ssim"] = {}
                for bayer_pattern, psnr_i, ssim_i in zip(bayer_patterns, psnr_o, ssim_o):
                    results[target_filename]["psnr"][bayer_pattern] = psnr_i.item()
                    results[target_filename]["ssim"][bayer_pattern] = ssim_i.item()
        if libvmaf_cuda:
            print(f"\nVMAF: {len(vmaf_cuda_tasks)}it on CUDA, {batch_count - len(vmaf_cuda_tasks)}it on CPU")
    except ExceptionGroup:
        print(f"\nError found in dataset near {target_filename}")
        raise
    # Ctrl + C
    except asyncio.exceptions.CancelledError:
        raise
    finally:
        with open(results_file, "w") as f:
            json.dump(results, f)
        print(f"\nResults written to {results_file}")

if __name__ == "__main__":
    asyncio.run(main())