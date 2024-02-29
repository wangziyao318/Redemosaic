import os
import json
import torch
import asyncio
from pathlib import Path
from tqdm.asyncio import tqdm
from skimage.io import imread_collection

from redemosaic import redemosaic
from image_metrics import peak_signal_noise_ratio, structural_similarity, multi_assessment_fusion

device = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")
"""
Given batch_size = len(bayer_patterns), hardware requirements for RAISE dataset:
    # redemosaic() + evaluate()
    torch.device("cuda") + {libvmaf_cuda, libvmaf} with batch_size 4 requires 16G+ VRAM depends on libvmaf_cuda_window_size. Slightly faster than the setting below for large images, and is equally fast for small images.
    torch.device("cuda") + libvmaf with batch size 4 requires >= 12G VRAM. Recommended for NVIDIA users and is at least 4 times faster than pure CPU.
    torch.device("cpu") + libvmaf with batch size 4 requires >= 16GB RAM and a powerful CPU. Not recommended because CPU doesn't benefit from batch_size.
    torch.device("cpu") + libvmaf with batch size 1 requires >= 8GB RAM. Recommended for low-end users. Do it 4 times with different Bayer pattern and set update_results to True.
"""
bayer_patterns = ["gbrg", "grbg", "bggr", "rggb"]
# bayer_patterns = ["gbrg"]
# bayer_patterns = ["grbg"]
# bayer_patterns = ["bggr"]
# bayer_patterns = ["rggb"]
"""
Set to True to update instead of overwritting the results file.
"""
update_results = False
"""
Default to False to delete redemosaiced images and metrics logs after computation.
Not recommended to switch. Intermediate files can take up to 4 times disk space compared with the input dataset.
"""
keep_temp_files = False
"""
Set VMAF model versions to use.
"""
vmaf_versions = ["vmaf_v0.6.1", "vmaf_4k_v0.6.1"]
"""
Set to True to use libvmaf_cuda, default to False to use libvmaf.
Require libvmaf compiled with -Denable_cuda=true and ffmpeg compiled with --enable-nonfree --enable-ffnvcodec --enable-libvmaf.
"""
libvmaf_cuda = False
"""
Set max concurrent libvmaf_cuda tasks on GPU, while extra tasks are assigned to CPU to balance system load and prevent CUDA out of memory.
Only works when libvmaf_cuda = True.
"""
libvmaf_cuda_window_size = 8
"""
Set relative directory for dataset and temporary storage.
target_dir is the directory of input dataset, and preds_dir temporarily holds redemosaiced images and metrics logs.
target_ext is the input image extension without dot, case sensitive.
Results is stored as a JSON file in current directory.
"""
target_dir = "real_img"
target_ext = "TIF"
# target_dir = "fake_img"
# target_ext = "png"
preds_dir = "temp"
results_filename = "results.json"
"""
Set where to start and how many to process, can be used as checkpoints.
The real batch_count is the minimum between the set value and number of remaining images.
"""
target_start_index = 0
batch_count = 1000

async def main():
    """
    Asynchronous main() function.
    """
    print(f"torch use {device}")
    print(f"ffmpeg use {'libvmaf_cuda' if libvmaf_cuda else 'libvmaf'}")
    if not os.path.exists(preds_dir):
        os.makedirs(preds_dir)

    targets = imread_collection(os.path.join(target_dir, "*." + target_ext), conserve_memory=True)
    target_paths = str(targets)[1:-1].translate({ord("'"): None}).split(", ")
    target_names = map(lambda e : Path(e).stem, target_paths)
    N = min(batch_count, len(target_paths) - target_start_index)

    results = {}
    if update_results:
        try:
            with open(results_filename, "r") as f:
                results = json.load(f)
        except OSError:
            raise
        except json.decoder.JSONDecodeError:
            print(f"\n{results_filename} corrupted, please set update_results to False to overwrite it.")
            return
    
    libvmaf_cuda_tasks = set()
    tasks_counter = [0, 0]
    try:
        """
        TaskGroup() is introduced in Python3.11, please upgrade Python if needed.
        """
        async with asyncio.TaskGroup() as tg:
            main_iterator = tqdm(zip(targets, target_names, range(N)), total=N, position=1)
            main_iterator.set_description("Main")
            libvmaf_iterator = tqdm(total=N, position=0)
            libvmaf_iterator.set_description("VMAF")
            async for target, target_name, _ in main_iterator:
                """
                Input image: target(H,W,3).
                """
                target = torch.tensor(target, dtype=torch.uint8, device=device)
                target_filename = target_name + "." + target_ext
                """
                Redemosaiced image: preds(B,H,W,3).
                """
                preds = redemosaic(target, bayer_patterns)

                if target_filename not in results.keys():
                    results[target_filename] = {}
                for metric in ["psnr", "ssim", "vmaf"]:
                    if metric not in results[target_filename].keys():
                        results[target_filename][metric] = {}
                """
                VMAF
                """
                if libvmaf_cuda and len(libvmaf_cuda_tasks) < libvmaf_cuda_window_size:
                    task = tg.create_task(
                        multi_assessment_fusion(
                            preds, preds_dir,
                            target_filename, target_dir,
                            results,
                            bayer_patterns, vmaf_versions,
                            True, libvmaf_iterator, keep_temp_files
                        )
                    )
                    libvmaf_cuda_tasks.add(task)
                    tasks_counter[0] += 1
                    task.add_done_callback(libvmaf_cuda_tasks.discard)
                else:
                    tg.create_task(
                        multi_assessment_fusion(
                            preds, preds_dir,
                            target_filename, target_dir,
                            results,
                            bayer_patterns, vmaf_versions,
                            False, libvmaf_iterator, keep_temp_files
                        )
                    )
                    tasks_counter[1] += 1
                await asyncio.sleep(0)
                """
                PSNR: psnr(B).
                """
                psnr_B = peak_signal_noise_ratio(preds, target)
                """
                SSIM: ssim(B).
                """
                ssim_B = structural_similarity(preds, target)

                for bayer_pattern, psnr, ssim in zip(bayer_patterns, psnr_B, ssim_B):
                    results[target_filename]["psnr"][bayer_pattern] = round(psnr.item(), 6)
                    results[target_filename]["ssim"][bayer_pattern] = round(ssim.item(), 6)
    except ExceptionGroup:
        print(f"\nError found in dataset around {os.path.join(target_dir, target_filename)}.")
        raise
    except asyncio.exceptions.CancelledError:
        raise
    finally:
        with open(results_filename, "w") as f:
            json.dump(results, f)
        print(f"\nVMAF:  {tasks_counter[0]}it on CUDA, {tasks_counter[1]}it on CPU.")
        print(f"Results written to {results_filename}.")

if __name__ == "__main__":
    asyncio.run(main())