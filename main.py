import skimage.io
import torch
from tqdm import tqdm

from redemosaic_fp16 import redemosaic

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
        redemosaic(torch.tensor(rgbimg, dtype=torch.uint8, device=device), device=device)
    
