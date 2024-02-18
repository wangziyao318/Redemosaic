import torch


def redemosaic(
        rgbimg: torch.Tensor,
        bayer_patterns: tuple[str] = ("gbrg", "grbg", "bggr", "rggb")
    ) -> torch.Tensor:
    """
    The function creates redemosaiced images of any Bayer patterns given rgbimage.
    
    Default Bayer patterns: ("gbrg", "grbg", "bggr", "rggb")

    Return: redemosaiced RGB images of B Bayer patterns (B, H, W, 3)
    """
    assert isinstance(rgbimg, torch.Tensor)
    assert isinstance(bayer_patterns, tuple)
    for bayer_pattern in bayer_patterns:
        assert bayer_pattern in ("gbrg", "grbg", "bggr", "rggb")
    device = rgbimg.device

    GR_GB = torch.tensor([
        [0, 0, -1, 0, 0],
        [0, 0, 2, 0, 0],
        [-1, 2, 4, 2, -1],
        [0, 0, 2, 0, 0],
        [0, 0, -1, 0, 0]
    ], dtype=torch.float32, device=device) / 8

    Rg_RB_Bg_BR = torch.tensor([
        [0, 0, 0.5, 0, 0],
        [0, -1, 0, -1, 0],
        [-1, 4, 5, 4, -1],
        [0, -1, 0, -1, 0],
        [0, 0, 0.5, 0, 0]
    ], dtype=torch.float32, device=device) / 8

    Rg_BR_Bg_RB = torch.t(Rg_RB_Bg_BR)

    Rb_BB_Br_RR = torch.tensor([
        [0, 0, -1.5, 0, 0],
        [0, 2, 0, 2, 0],
        [-1.5, 0, 6, 0, -1.5],
        [0, 2, 0, 2, 0],
        [0, 0, -1.5, 0, 0]
    ], dtype=torch.float32, device=device) / 8
    
    basic_masks = torch.zeros([4, rgbimg.size(0), rgbimg.size(1)], dtype=torch.bool, device=device)
    for basic_mask, (i, j) in zip(basic_masks, [(0, 0), (0, 1), (1, 0), (1, 1)]):
        for x in basic_mask[i::2]: x[j::2] = 1

    rgbmasks_bayerpatterns = torch.zeros([4, 3, rgbimg.size(0), rgbimg.size(1)], dtype=torch.bool, device=device)
    rgb_bayerpatterns = torch.zeros_like(rgbmasks_bayerpatterns, dtype=torch.float32, device=device)
    for rgbmasks, rgb, bayer_pattern in zip(rgbmasks_bayerpatterns, rgb_bayerpatterns, bayer_patterns):
        rgbmasks[0] = basic_masks[bayer_pattern.find("r")]
        rgb[0] = rgbimg[:,:,0] * rgbmasks[0]
        rgbmasks[1] = (basic_masks[bayer_pattern.find("g")] + basic_masks[bayer_pattern.rfind("g")])
        rgb[1] = rgbimg[:,:,1] * rgbmasks[1]
        rgbmasks[2] = basic_masks[bayer_pattern.find("b")]
        rgb[2] = rgbimg[:,:,2] * rgbmasks[2]
    cfa_bayerpatterns = torch.sum(rgb_bayerpatterns, dim=1, keepdim=True)

    del basic_masks

    # Gradient-corrected bilinear interpolated G at all R and B.
    rgb_bayerpatterns[:,1,:,:] = torch.where(
        torch.logical_or(rgbmasks_bayerpatterns[:,0,:,:] == 1, rgbmasks_bayerpatterns[:,2,:,:] == 1),
        torch.conv2d(input=torch.nn.ReflectionPad2d(2)(cfa_bayerpatterns), weight=GR_GB[None, None, ...]).squeeze(1),
        rgb_bayerpatterns[:,1,:,:]
    )

    RBg_RBBR = torch.conv2d(torch.nn.ReflectionPad2d(2)(cfa_bayerpatterns), Rg_RB_Bg_BR[None, None, ...]).squeeze(1)
    RBg_BRRB = torch.conv2d(torch.nn.ReflectionPad2d(2)(cfa_bayerpatterns), Rg_BR_Bg_RB[None, None, ...]).squeeze(1)
    RBgr_BBRR = torch.conv2d(torch.nn.ReflectionPad2d(2)(cfa_bayerpatterns), Rb_BB_Br_RR[None, None, ...]).squeeze(1)

    del GR_GB, Rg_RB_Bg_BR, Rg_BR_Bg_RB, Rb_BB_Br_RR, cfa_bayerpatterns

    R_row = torch.any(rgbmasks_bayerpatterns[:,0,:,:] == 1, axis=2).unsqueeze(2) * torch.ones_like(rgb_bayerpatterns[:,0,:,:], dtype=torch.bool, device=device)
    R_col = torch.any(rgbmasks_bayerpatterns[:,0,:,:] == 1, axis=1).unsqueeze(1) * torch.ones_like(rgb_bayerpatterns[:,0,:,:], dtype=torch.bool, device=device)

    B_row = torch.any(rgbmasks_bayerpatterns[:,2,:,:] == 1, axis=2).unsqueeze(2) * torch.ones_like(rgb_bayerpatterns[:,2,:,:], dtype=torch.bool, device=device)
    B_col = torch.any(rgbmasks_bayerpatterns[:,2,:,:] == 1, axis=1).unsqueeze(1) * torch.ones_like(rgb_bayerpatterns[:,2,:,:], dtype=torch.bool, device=device)

    rgb_bayerpatterns[:,0,:,:] = torch.where(torch.logical_and(R_row == 1, B_col == 1), RBg_RBBR, rgb_bayerpatterns[:,0,:,:])
    rgb_bayerpatterns[:,0,:,:] = torch.where(torch.logical_and(B_row == 1, R_col == 1), RBg_BRRB, rgb_bayerpatterns[:,0,:,:])

    rgb_bayerpatterns[:,2,:,:] = torch.where(torch.logical_and(B_row == 1, R_col == 1), RBg_RBBR, rgb_bayerpatterns[:,2,:,:])
    rgb_bayerpatterns[:,2,:,:] = torch.where(torch.logical_and(R_row == 1, B_col == 1), RBg_BRRB, rgb_bayerpatterns[:,2,:,:])

    rgb_bayerpatterns[:,0,:,:] = torch.where(torch.logical_and(B_row == 1, B_col == 1), RBgr_BBRR, rgb_bayerpatterns[:,0,:,:])
    rgb_bayerpatterns[:,2,:,:] = torch.where(torch.logical_and(R_row == 1, R_col == 1), RBgr_BBRR, rgb_bayerpatterns[:,2,:,:])

    del RBg_RBBR, RBg_BRRB, RBgr_BBRR, R_row, R_col, B_row, B_col, rgbmasks_bayerpatterns

    return torch.stack([torch.clamp(rgb_bayerpatterns[:,0,:,:], 0, 255),
                        torch.clamp(rgb_bayerpatterns[:,1,:,:], 0, 255),
                        torch.clamp(rgb_bayerpatterns[:,2,:,:], 0, 255)], 3).type(torch.uint8)