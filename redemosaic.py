from skimage import io
import torch
import time

# return RGB mask given pattern
def bayer_mask(rgbimgs, patterns):

    masks = torch.zeros([len(rgbimgs), len(patterns, )], dtype=torch.bool)

    for rgbimg in rgbimgs:

        print("hello")



    # # init dict object channels contain "r", "g", and "b" with same input shape
    # channels = {channel: torch.zeros([size[0], size[1]], dtype=torch.bool, device=device) for channel in "rgb"}

    # for channel, (y, x) in zip(pattern, [(0, 0), (0, 1), (1, 0), (1, 1)]):
    #     # not a good implementation, but channels[channel][y::2][x::2]=1 not work
    #     for col in channels[channel][y::2]:
    #         col[x::2] = 1

    # return tuple(channels.values())
    return 0,1,2

# mosaic
def mosaic(rgbimg, R_m, G_m, B_m):
    return rgbimg[:,:,0] * R_m + rgbimg[:,:,1] * G_m + rgbimg[:,:,2] * B_m

# demosaic
def demosaic(cfaimg, R_m, G_m, B_m):
    # [None, None, ...] equivalent to .unsqueeze(0).unsqueeze(0)
    cfaimg = cfaimg.float()[None, None, ...]

    GR_GB = torch.tensor(
        [
            [0, 0, -1, 0, 0],
            [0, 0, 2, 0, 0],
            [-1, 2, 4, 2, -1],
            [0, 0, 2, 0, 0],
            [0, 0, -1, 0, 0],
        ], dtype=torch.float32, device=device
    ) / 8

    Rg_RB_Bg_BR = torch.tensor(
        [
            [0, 0, 0.5, 0, 0],
            [0, -1, 0, -1, 0],
            [-1, 4, 5, 4, -1],
            [0, -1, 0, -1, 0],
            [0, 0, 0.5, 0, 0],
        ], dtype=torch.float32, device=device
    ) / 8

    Rg_BR_Bg_RB = torch.t(Rg_RB_Bg_BR)

    Rb_BB_Br_RR = torch.tensor(
        [
            [0, 0, -1.5, 0, 0],
            [0, 2, 0, 2, 0],
            [-1.5, 0, 6, 0, -1.5],
            [0, 2, 0, 2, 0],
            [0, 0, -1.5, 0, 0],
        ], dtype=torch.float32, device=device
    ) / 8

    R = cfaimg * R_m[None, None, ...]
    G = cfaimg * G_m[None, None, ...]
    B = cfaimg * B_m[None, None, ...]

    # calculate bilinear G value at all R and B locations
    G = torch.where(torch.logical_or(R_m == 1, B_m == 1), torch.conv2d(torch.nn.ReflectionPad2d(2)(cfaimg), GR_GB[None, None, ...]), G)

    RBg_RBBR = torch.conv2d(torch.nn.ReflectionPad2d(2)(cfaimg), Rg_RB_Bg_BR[None, None, ...])
    RBg_BRRB = torch.conv2d(torch.nn.ReflectionPad2d(2)(cfaimg), Rg_BR_Bg_RB[None, None, ...])
    RBgr_BBRR = torch.conv2d(torch.nn.ReflectionPad2d(2)(cfaimg), Rb_BB_Br_RR[None, None, ...])

    del GR_GB, Rg_RB_Bg_BR, Rg_BR_Bg_RB, Rb_BB_Br_RR

    # Red rows.
    R_r = torch.t(torch.any(R_m == 1, axis=1)[None]) * torch.ones(R.size(), device=device)
    # Red columns.
    R_c = torch.any(R_m == 1, axis=0)[None] * torch.ones(R.size(), device=device)
    # Blue rows.
    B_r = torch.t(torch.any(B_m == 1, axis=1)[None]) * torch.ones(B.size(), device=device)
    # Blue columns
    B_c = torch.any(B_m == 1, axis=0)[None] * torch.ones(B.size(), device=device)

    R = torch.where(torch.logical_and(R_r == 1, B_c == 1), RBg_RBBR, R)
    R = torch.where(torch.logical_and(B_r == 1, R_c == 1), RBg_BRRB, R)

    B = torch.where(torch.logical_and(B_r == 1, R_c == 1), RBg_RBBR, B)
    B = torch.where(torch.logical_and(R_r == 1, B_c == 1), RBg_BRRB, B)

    R = torch.where(torch.logical_and(B_r == 1, B_c == 1), RBgr_BBRR, R)
    B = torch.where(torch.logical_and(R_r == 1, R_c == 1), RBgr_BBRR, B)

    del RBg_RBBR, RBg_BRRB, RBgr_BBRR, R_r, R_c, B_r, B_c

    # remove values out of bonds (0-255)
    return torch.stack([torch.clamp(R.squeeze([0, 1]), 0, 255),
                        torch.clamp(G.squeeze([0, 1]), 0, 255),
                        torch.clamp(B.squeeze([0, 1]), 0, 255)], 2).type(torch.uint8)



if __name__ == "__main__":
    start = time.time()

    # enable gpu acceleration if available
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")
    print(f"torch use {device}")

    # sensor alignments tuple
    patterns = ["gbrg", "grbg", "bggr", "rggb"]
    print(f"sensor alignment patterns: {patterns}")

    # torch can't read or write tif, use scikit-image helper
    # rgbimg collection
    rgbimgs = io.ImageCollection("img/*.TIF", load_func=io.imread)
    print(f"number of input image: {len(rgbimgs)}")
    print("scimageread\t{:.3f}s".format(time.time()-start))

    # calculate RGB mask to reuse in both mosaic() and demosaic()
    r_ms, g_ms, b_ms = bayer_mask(rgbimgs, patterns)

    # cfaimg = mosaic(rgbimg, R_m, G_m, B_m)
    # print("mosaic\t\t{:.3f}s".format(time.time()-start))

    # newimg = demosaic(cfaimg, R_m, G_m, B_m)
    # print("demosaic\t{:.3f}s".format(time.time()-start))

    # pyvips.Image.new_from_array(newimg.cpu().detach().numpy(), interpretation="rgb").write_to_file("new-cv.tif")
    # print("pyvipswrite\t{:.3f}s".format(time.time()-start))