import torch


def PSNR(
        preds: torch.Tensor,
        targets: torch.Tensor,
        data_range: int = 255
    ) -> torch.Tensor:
    assert isinstance(data_range, int)
    device = preds.device

    MAX_i = torch.tensor(data_range, dtype=torch.float32, device=device)

    # use torch.mean()

    MSE = torch.sum(torch.pow(preds - targets, 2)) / torch.tensor(targets.numel(), dtype=torch.float32, device=device)
    psnr_pred_target = 10 * (2 * torch.log10(MAX_i) - torch.log10(MSE))

    return psnr_pred_target


def SSIM(
        preds: torch.Tensor,
        targets: torch.Tensor,
        data_range: int = 255,
        K1: float = 0.01,
        K2: float = 0.03,
) -> torch.Tensor:
    
    """
    input: (4, L, W, 3)
    Return: (4)
    """

    device = preds.device

    if preds.size() != targets.size():
        raise RuntimeError(
            f"Predictions and targets are expected to have the same shape, but got {preds.size()} and {targets.size()}."
        )

    # use ndim

    if len(preds.size()) != 4:
        raise ValueError(
            "Expected `preds` and `target` to have BxHxWxC shape."
            f" Got preds: {preds.size()} and target: {targets.size()}."
        )

    # NOT tensor
    C1 = pow(K1 * data_range, 2)
    C2 = pow(K2 * data_range, 2)

    # may pad pred and target for conv2d


    # Y = 0.2989 * R + 0.5870 * G + 0.1140 * B
    rgb2y = torch.tensor([0.2989, 0.5870, 0.1140])


    # use float32 if possible
    preds_y = (preds @ rgb2y).float()
    targets_y = (targets @ rgb2y).float()

    # window size for uniform filter
    win_size = 7


    # Y (L, W) for each Bayer pattern (4, L, W)
    for pred_y, target_y in zip(preds_y, targets_y):

        kernel = torch.ones((1,1,7,7), dtype=torch.float32) / torch.prod(
            torch.tensor([7,7], dtype=torch.float32)
        )

        # scikit-image use symmetric reflection padding, which is with repeat of boundary; but torch does not
        ux = torch.nn.functional.conv2d(torch.nn.ReflectionPad2d(3)(pred_y[None, None, ...]), kernel)
        uy = torch.nn.functional.conv2d(torch.nn.ReflectionPad2d(3)(target_y[None, None, ...]), kernel)

        uxx = torch.nn.functional.conv2d(torch.nn.ReflectionPad2d(3)((pred_y * pred_y)[None, None, ...]), kernel)
        uyy = torch.nn.functional.conv2d(torch.nn.ReflectionPad2d(3)((target_y * target_y)[None, None, ...]), kernel)
        uxy = torch.nn.functional.conv2d(torch.nn.ReflectionPad2d(3)((pred_y * target_y)[None, None, ...]), kernel)


        vx = uxx - ux * ux
        vy = uyy - uy * uy
        vxy = uxy - ux * uy

        A1, A2, B1, B2 = ((2 * ux * uy + C1,
                       2 * vxy + C2,
                       ux ** 2 + uy ** 2 + C1,
                       vx + vy + C2))

        D = B1 * B2
        S = (A1 * A2) / D

        ssim = S[..., 3:-3, 3:-3]


    return torch.mean(ssim, dtype=torch.float64)