import os
import json
import torch
import asyncio
from pathlib import Path
from tqdm.asyncio import tqdm
from skimage.io import imsave

def psnr_rgb(
        preds: torch.Tensor,
        target: torch.Tensor,
        data_range: int = 255
    ) -> torch.Tensor:
    """
    The function calculates PSNR of R,G,B channels between B predictions and the target.

    Input: preds(B, H, W, 3) and target(H, W, 3)

    Return: (B, 3)
    """
    assert preds.ndim == 4
    B = preds.size(0)
    
    MSE = torch.mean(torch.pow(preds - target.expand(B, -1, -1, -1), 2),
                     dim=(1,2), keepdim=True, dtype=torch.float32).squeeze((1,2))
    return 10 * (2 * torch.log10(torch.full((B, 3), data_range, dtype=torch.float32, device=preds.device)) - torch.log10(MSE))

async def _metrics_compute(
        preds_path: str,
        target_path: str,
        batch_size: int,
        vmaf_versions: list[str] = ["vmaf_v0.6.1"],
        libvmaf_cuda: bool = False
    ) -> list[list[float]]:
    """
    This async function uses libvmaf or libvmaf_cuda to calculate PSNR_Y, SSIM, and VMAF between B predictions and the target given vmaf_versions.

    It is a FFmpeg wrapper and requires ffmpeg binary in path.

    Return: [[_ * B], [_ * B], [[_ * B] * len(vmaf_versions)]]
    """
    vmaf_versions_str = "|".join(["version=" + vmaf_version + "\\\\:name=" + vmaf_version for vmaf_version in vmaf_versions])
    target_name = Path(target_path).stem
    preds_dir = os.path.dirname(preds_path)
    log_path = os.path.join(preds_dir, target_name + ".json")
    
    if libvmaf_cuda:
        proc = await asyncio.create_subprocess_shell(
            f'''ffmpeg \
            -framerate 1 -i {preds_path} \
            -i {target_path} \
            -lavfi "
                format=yuv420p,hwupload_cuda,scale_cuda[pred]; \
                format=yuv420p,hwupload_cuda,scale_cuda[target]; \
                [pred][target]libvmaf_cuda='model={vmaf_versions_str}:feature=name=psnr|name=float_ssim:log_path={log_path}:log_fmt=json'
            " \
            -f null -''',
            stderr=asyncio.subprocess.PIPE)
    else:
        proc = await asyncio.create_subprocess_shell(
            f'''ffmpeg \
            -framerate 1 -i {preds_path} \
            -i {target_path} \
            -lavfi libvmaf='model={vmaf_versions_str}:feature=name=psnr|name=float_ssim:log_path={log_path}:log_fmt=json:n_threads={batch_size * 2}' \
            -f null -''',
            stderr=asyncio.subprocess.PIPE)
    _, stderr = await proc.communicate()

    try:
        with open(log_path, "r") as f:
            log = json.load(f)
    except OSError:
        print(stderr.decode())
        raise

    results = [[float(frame["metrics"]["psnr_y"]), float(frame["metrics"]["float_ssim"]), [float(frame["metrics"][vmaf_version])
                    for vmaf_version in vmaf_versions]]
                    for frame in log["frames"]]
    results = [list(i) for i in zip(*results)]
    results[2] = [list(i) for i in zip(*results[2])]
    return results

async def evaluate(
        preds: torch.Tensor,
        preds_dir: str,
        target_filename: str,
        target_dir: str,
        results: dict,
        bayer_patterns: list[str],
        vmaf_versions: list[str] = ["vmaf_v0.6.1"],
        libvmaf_cuda: bool = False,
        tqdm_iterator: tqdm = None,
        keep_temp_files: bool = False
    ):
    """
    This async function computes PSNR_Y, SSIM, and VMAF between B predictions and the target given vmaf_versions.

    It saves predictions and results logs as temporary files and deletes them after.

    Results are written to dict so there's no return value.
    """
    B = len(bayer_patterns)
    assert preds.size(0) == B
    for vmaf_version in vmaf_versions:
        assert vmaf_version in ["vmaf_v0.6.1", "vmaf_4k_v0.6.1"]
    target_name = Path(target_filename).stem

    preds_paths = [os.path.join(preds_dir, str(i+1) + "_" + target_name + ".TIF") for i in range(B)]
    for pred, pred_path in zip(preds, preds_paths):
        imsave(pred_path, pred.cpu().detach().numpy(), plugin="tifffile", check_contrast=False)

    metrics = await _metrics_compute(
        os.path.join(preds_dir, "%d_" + target_name + ".TIF"),
        os.path.join(target_dir, target_filename),
        B, vmaf_versions, libvmaf_cuda)
    
    if not keep_temp_files:
        await asyncio.create_subprocess_shell(f'''rm {" ".join(preds_paths)} {os.path.join(preds_dir, target_name + ".json")}''')

    if "Y" not in results[target_filename]["psnr"].keys():
        results[target_filename]["psnr"]["Y"] = {}
    for bayer_pattern, psnr_i, ssim_i in zip(bayer_patterns, metrics[0], metrics[1]):
        results[target_filename]["psnr"]["Y"][bayer_pattern] = psnr_i
        results[target_filename]["ssim"][bayer_pattern] = ssim_i
    for vmaf_version, vmaf_l in zip(vmaf_versions, metrics[2]):
        if vmaf_version not in results[target_filename]["vmaf"].keys():
            results[target_filename]["vmaf"][vmaf_version] = {}
        for bayer_pattern, vmaf_i in zip(bayer_patterns, vmaf_l):
            results[target_filename]["vmaf"][vmaf_version][bayer_pattern] = vmaf_i
    
    if tqdm_iterator is not None:
        tqdm_iterator.update(1)