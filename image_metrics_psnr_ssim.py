import torch
import asyncio
from torchvision.transforms.functional import pad


def psnr(
        preds: torch.Tensor,
        target: torch.Tensor,
        data_range: int = 255
    ) -> torch.Tensor:
    """
    The function calculates PSNR between batch of predictions and target given data range.

    Input: preds(B, H, W, 3) and target(H, W, 3)

    Return: psnr(B)
    """
    assert preds.ndim == 4
    B = preds.size(0)
    
    MSE = torch.mean(torch.pow(preds - target.expand(B, -1, -1, -1), 2),
                     dim=(1,2,3), keepdim=True, dtype=torch.float32).squeeze((1,2,3))
    return 10 * (2 * torch.log10(torch.full((B,), data_range, dtype=torch.float32, device=preds.device)) - torch.log10(MSE))


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

    Scikit-learn uses symmetric padding while torch uses reflection padding, making results vary slightly.
    
    From Rec.709, Y = 0.2989 * R + 0.5870 * G + 0.1140 * B

    window_size: length of convolution kernel, default 7 gives padding 3, must be odd number.

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


async def vmaf():

    return 0