import os
import json
import torch
import asyncio
from pathlib import Path
from tqdm.asyncio import tqdm
from skimage.io import imsave
from torchvision.transforms.functional import pad

def peak_signal_noise_ratio(
        preds: torch.Tensor,
        target: torch.Tensor,
        data_range: int = 255
    ) -> torch.Tensor:
    """
    The function calculates mean PSNR of RGB channels between B predictions and the target.

    data_range: 255 for uint8, 1 for float.

    Input: preds(B, H, W, 3) and target(H, W, 3)

    Return: psnr(B)
    """
    assert preds.ndim == 4
    B = preds.size(0)
    
    MSE = torch.mean(torch.pow(preds.float() - target.float().expand(B, -1, -1, -1), 2),
                     dim=(1,2,3), dtype=torch.float32)
    return 10 * (2 * torch.log10(torch.full((B,), data_range, dtype=torch.float32, device=preds.device)) - torch.log10(MSE))

def structural_similarity(
        preds: torch.Tensor,
        target: torch.Tensor,
        data_range: int = 255,
        window_size: int = 7,
        K1: float = .01,
        K2: float = .03
    ) -> torch.Tensor:
    """
    The function calculates mean SSIM on RGB channels between batch of predictions and target given data range.

    It uses symmetric padding to be consistent with scikit-image.

    data_range: 255 for uint8, 1 for float.

    window_size: length of convolution kernel, default 7 gives padding 3, must be odd number. Different window_size produces different results.

    Input: preds(B, H, W, 3) and target(H, W, 3)

    Return: ssim(B)
    """
    assert preds.ndim == 4
    assert window_size % 2 == 1
    device = preds.device
    B = preds.size(0)
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2
    NP = window_size * window_size
    cov_norm = NP / (NP-1.)
    padding = (window_size - 1) // 2

    preds = preds.float().permute(3,0,1,2)
    targets = target.float().permute(2,0,1).unsqueeze(1).expand(-1, B, -1, -1)
    kernel = torch.full((1, 1, window_size, window_size), 1./NP, dtype=torch.float32, device=device)

    del target, NP

    results = []
    for preds_c, targets_c in zip(preds, targets):
        inputs = torch.cat((preds_c,
                            targets_c,
                            preds_c * preds_c,
                            targets_c * targets_c,
                            preds_c * targets_c)).unsqueeze(1)
        ux, uy, uxx, uyy, uxy = torch.conv2d(pad(inputs, padding, padding_mode="symmetric"), kernel).split(B)

        del inputs

        vx = cov_norm * (uxx - ux * ux)
        vy = cov_norm * (uyy - uy * uy)
        vxy = cov_norm * (uxy - ux * uy)

        del uxx, uyy, uxy
        
        S = (2 * ux * uy + C1) * (2 * vxy + C2) / ((ux * ux + uy * uy + C1) * (vx + vy + C2))

        results.append(torch.mean(S[..., padding:-padding, padding:-padding], dim=(1,2,3), dtype=torch.float32))

        del ux, uy, vx, vy, vxy, S

    return torch.mean(torch.stack(results), dim=0, dtype=torch.float32)

async def _vmaf_compute(
        preds_path: str,
        target_path: str,
        batch_size: int,
        vmaf_versions: list[str] = ["vmaf_v0.6.1"],
        libvmaf_cuda: bool = False
    ) -> list[list[float]]:
    """
    This async function uses libvmaf or libvmaf_cuda to calculate VMAF between B predictions and the target given vmaf_versions.

    It is a FFmpeg wrapper and requires ffmpeg binary in path.

    Return: [[_ * B] * len(vmaf_versions)]
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
            log = json.load(f)
    except OSError:
        print(stderr.decode())
        raise

    results = [[float(frame["metrics"][vmaf_version]) for vmaf_version in vmaf_versions] for frame in log["frames"]]

    return [list(i) for i in zip(*results)]

async def multi_assessment_fusion(
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
    This async function calculates VMAF between B predictions and the target given vmaf_versions.

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

    vmaf_BV = await _vmaf_compute(
        os.path.join(preds_dir, "%d_" + target_name + ".TIF"),
        os.path.join(target_dir, target_filename),
        B, vmaf_versions, libvmaf_cuda)
    
    if not keep_temp_files:
        await asyncio.create_subprocess_shell(f'''rm {" ".join(preds_paths)} {os.path.join(preds_dir, target_name + ".json")}''')

    for vmaf_version, vmaf_B in zip(vmaf_versions, vmaf_BV):
        if vmaf_version not in results[target_filename]["vmaf"].keys():
            results[target_filename]["vmaf"][vmaf_version] = {}
        for bayer_pattern, vmaf in zip(bayer_patterns, vmaf_B):
            results[target_filename]["vmaf"][vmaf_version][bayer_pattern] = vmaf
    
    if tqdm_iterator is not None:
        tqdm_iterator.update(1)