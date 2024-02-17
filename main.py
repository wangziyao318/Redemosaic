import skimage.io
import torch
from tqdm import tqdm

from redemosaic import redemosaic

device = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")

bayer_patterns = ["gbrg", "grbg", "bggr", "rggb"]

if __name__ == "__main__":
    print(f"torch use {device}")

    rgbimgs = skimage.io.imread_collection("img/*.TIF", conserve_memory=True)
    # imgpaths = str(rgbimgs)[1:-1].split(', ')
    print(f"number of input images: {len(rgbimgs)}")

    for rgbimg in tqdm(rgbimgs, position=0):
        print(redemosaic(torch.tensor(rgbimg, dtype=torch.uint8, device=device), device=device).size())
    

    # calculate RGB mask to reuse in both mosaic() and demosaic()
    # r_ms, g_ms, b_ms = bayer_mask(rgbimgs, patterns)

    # cfaimg = mosaic(rgbimg, R_m, G_m, B_m)
    # print("mosaic\t\t{:.3f}s".format(time.time()-start))

    # newimg = demosaic(cfaimg, R_m, G_m, B_m)
    # print("demosaic\t{:.3f}s".format(time.time()-start))

    # pyvips.Image.new_from_array(newimg.cpu().detach().numpy(), interpretation="rgb").write_to_file("new-cv.tif")
    # print("pyvipswrite\t{:.3f}s".format(time.time()-start))
    
    # if timer: print("scimageread\t{:.3f}s".format(time.process_time()-begin_time))