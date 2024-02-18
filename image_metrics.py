import torch
from torchvision.transforms.functional import pad


def psnr(
        preds: torch.Tensor,
        target: torch.Tensor,
        data_range: float = 255.
    ) -> torch.Tensor:
    """
    The function calculates PSNR between batch of predictions and target given data range.

    Input: preds(B, H, W, 3) and target(H, W, 3)

    Return: psnr(B)
    """
    assert preds.ndim == 4
    assert data_range >= 0 and data_range <= 255
    targets = target.unsqueeze(0).repeat(preds.size(0), 1, 1, 1)

    MSE = torch.mean((preds - targets) ** 2, dim=(1,2,3), keepdim=True, dtype=torch.float32).squeeze((1,2,3))
    return 10 * (2 * torch.log10(torch.ones_like(MSE, dtype=torch.float32, device=preds.device) * data_range) - torch.log10(MSE))


def ssim(
        preds: torch.Tensor,
        target: torch.Tensor,
        data_range: float = 255.,
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
    assert data_range >= 0 and data_range <= 255
    assert window_size % 2 == 1
    device = preds.device
    padding = window_size // 2
    targets = target.unsqueeze(0).repeat(preds.size(0), 1, 1, 1)

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2
    NP = window_size * window_size
    RGB2Y = torch.tensor((0.2989, 0.5870, 0.1140), dtype=torch.float32, device=device)
    
    preds_y = preds.float() @ RGB2Y
    targets_y = targets.float() @ RGB2Y
    cov_norm = NP / (NP-1.)
    kernel = torch.ones((1, 1, window_size, window_size), dtype=torch.float32, device=device) / NP

    inputs = torch.cat((preds_y.unsqueeze(1), targets_y.unsqueeze(1), (preds_y * preds_y).unsqueeze(1), (targets_y * targets_y).unsqueeze(1), (preds_y * targets_y).unsqueeze(1)))
    ux, uy, uxx, uyy, uxy = torch.conv2d(pad(inputs, padding, padding_mode="symmetric"), kernel).split(preds_y.size(0))

    del preds_y, targets_y, kernel

    vx = cov_norm * (uxx - ux * ux)
    vy = cov_norm * (uyy - uy * uy)
    vxy = cov_norm * (uxy - ux * uy)
    
    A1 = 2 * ux * uy + C1
    A2 = 2 * vxy + C2
    B1 = ux * ux + uy * uy + C1
    B2 = vx + vy + C2
    D = B1 * B2
    S = (A1 * A2) / D

    return torch.mean(S[..., padding:-padding, padding:-padding], dim=(1,2,3), keepdim=True, dtype=torch.float32).squeeze((1,2,3))