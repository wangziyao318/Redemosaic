import os
import json
import torch
import asyncio
from pathlib import Path
from tqdm.asyncio import tqdm
from skimage.io import imsave
from torchvision.transforms.functional import pad

'''
These functions are implemented but not used.
'''

def ssim(
        preds: torch.Tensor,
        target: torch.Tensor,
        data_range: int = 255,
        window_size: int = 7,
        K1: float = .01,
        K2: float = .03
    ) -> torch.Tensor:
    """
    The function calculates SSIM on Y between batch of predictions and target given data range.

    Some algorithm uses mean of SSIM on R,G,B, which produces different results.

    It uses symmetric padding to be consistent with scikit-image.
    
    From Rec.709, Y = 0.2989 * R + 0.5870 * G + 0.1140 * B. Different standard produces different results.

    window_size: length of convolution kernel, default 7 gives padding 3, must be odd number. Different window_size produces different results.

    Input: preds(B, H, W, 3) and target(H, W, 3)

    Return: ssim(B)
    """
    assert preds.ndim == 4
    assert window_size % 2 == 1
    device = preds.device
    B = preds.size(0)
    padding = window_size // 2

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2
    NP = window_size * window_size
    RGB2Y = torch.tensor((0.2989, 0.5870, 0.1140), dtype=torch.float32, device=device)
    
    cov_norm = NP / (NP-1.)
    preds_y = torch.matmul(preds.float(), RGB2Y)
    targets_y = torch.matmul(target.float(), RGB2Y).expand(B, -1, -1)
    
    kernel = torch.full((1, 1, window_size, window_size), 1./NP, dtype=torch.float32, device=device)

    del preds, target, NP, RGB2Y

    inputs = torch.cat((preds_y,
                        targets_y,
                        preds_y * preds_y,
                        targets_y * targets_y,
                        preds_y * targets_y)).unsqueeze(1)
    ux, uy, uxx, uyy, uxy = torch.conv2d(pad(inputs, padding, padding_mode="symmetric"), kernel).split(B)

    del preds_y, targets_y, kernel, inputs

    vx = cov_norm * (uxx - ux * ux)
    vy = cov_norm * (uyy - uy * uy)
    vxy = cov_norm * (uxy - ux * uy)

    S = (2 * ux * uy + C1) * (2 * vxy + C2) / ((ux * ux + uy * uy + C1) * (vx + vy + C2))

    del cov_norm, ux, uy, uxx, uyy, uxy, vx, vy, vxy

    return torch.mean(S[..., padding:-padding, padding:-padding], dim=(1,2,3), keepdim=True, dtype=torch.float32).squeeze((1,2,3))



async def _vmaf_compute(
        preds_path: str,
        target_path: str,
        batch_size: int,
        vmaf_versions: list[str] = ["vmaf_v0.6.1"],
        libvmaf_cuda: bool = False
    ) -> list[list[float]]:
    """
    This async function uses CPU or CUDA to calculate VMAF between batch of prediction images and target image given vmaf_versions.

    It is just a ffmpeg wrapper and requires libvmaf-enabled ffmpeg in path.

    It is intended for asyncio task instantiation.
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
            -filter_complex "
                format=yuv420p,hwupload_cuda,scale_cuda[pred]; \
                format=yuv420p,hwupload_cuda,scale_cuda[target]; \
                [pred][target]libvmaf_cuda='model={vmaf_versions_str}:log_path={log_path}:log_fmt=json'
            " \
            -f null -''',
            stderr=asyncio.subprocess.PIPE)
    else:
        proc = await asyncio.create_subprocess_shell(
            f'''ffmpeg \
            -framerate 1 -i {preds_path} \
            -i {target_path} \
            -lavfi libvmaf='model={vmaf_versions_str}:log_path={log_path}:log_fmt=json:n_threads={batch_size * 2}' \
            -f null -''',
            stderr=asyncio.subprocess.PIPE)
    _, stderr = await proc.communicate()

    try:
        with open(log_path, "r") as f:
            logs = json.load(f)
    except OSError:
        print(stderr.decode())
        raise
    except json.decoder.JSONDecodeError:
        print(stderr.decode())
        raise

    results = [[float(frame["metrics"][vmaf_version]) for vmaf_version in vmaf_versions] for frame in logs["frames"]]
    # transpose results
    return [list(i) for i in zip(*results)]


async def vmaf(
        preds: torch.Tensor,
        preds_dir: str,
        target_name: str,
        target_ext: str,
        target_dir: str,
        results: dict,
        bayer_patterns: list[str],
        vmaf_versions: list[str] = ["vmaf_v0.6.1"],
        libvmaf_cuda: bool = False,
        tqdm_iterator: tqdm = None
    ):
    """
    This async function computes VMAF between batch of predictions and target using different versions of VMAF models.

    It saves predictions and results as files and deletes them after.

    VMAF computation is done asynchronously to make full use of CPU.
    """
    assert preds.size(0) == len(bayer_patterns)
    for vmaf_version in vmaf_versions:
        assert vmaf_version in ["vmaf_v0.6.1", "vmaf_4k_v0.6.1"]
    batch_size = len(bayer_patterns)
    target_filename = target_name + "." + target_ext

    preds_files = [os.path.join(preds_dir, str(i+1) + "_" + target_filename) for i in range(batch_size)]
    for pred, pred_file in zip(preds, preds_files):
        imsave(pred_file, pred.cpu().detach().numpy(), check_contrast=False)

    vmafs = await _vmaf_compute(
        os.path.join(preds_dir, "%d_" + target_filename),
        os.path.join(target_dir, target_filename),
        batch_size, vmaf_versions, libvmaf_cuda)

    await asyncio.create_subprocess_shell(f'''rm {" ".join(preds_files)} {os.path.join(preds_dir, target_name + ".json")}''')

    if "vmaf" not in results[target_filename].keys():
        results[target_filename]["vmaf"] = {}
    for vmaf_version, vmaf_l in zip(vmaf_versions, vmafs):
        if vmaf_version not in results[target_filename]["vmaf"].keys():
            results[target_filename]["vmaf"][vmaf_version] = {}
        for bayer_pattern, vmaf_i in zip(bayer_patterns, vmaf_l):
            results[target_filename]["vmaf"][vmaf_version][bayer_pattern] = vmaf_i
    
    if tqdm_iterator is not None:
        tqdm_iterator.update(1)