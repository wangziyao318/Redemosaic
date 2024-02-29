import json
import matplotlib.pyplot as plt



if __name__ == "__main__":
    try:
        with open("results_real.json", "r") as f:
            results_real = json.load(f)
    except OSError:
        raise
    except json.decoder.JSONDecodeError:
        raise

    if results_real == {}:
        print("empty results real")
        exit

    try:
        with open("results_fake.json", "r") as f:
            results_fake = json.load(f)
    except OSError:
        raise
    except json.decoder.JSONDecodeError:
        raise

    if results_fake == {}:
        print("empty results fake")
        exit
    

    psnrs_real = [max(results_real[img]["psnr"].values()) for img in results_real]
    ssims_real = [max(results_real[img]["ssim"].values()) for img in results_real]
    vmafs_FHD_real = [max(results_real[img]["vmaf"]["vmaf_v0.6.1"].values()) for img in results_real]
    vmafs_4K_real = [max(results_real[img]["vmaf"]["vmaf_4k_v0.6.1"].values()) for img in results_real]

    psnrs_fake = [max(results_fake[img]["psnr"].values()) for img in results_fake]
    ssims_fake = [max(results_fake[img]["ssim"].values()) for img in results_fake]
    vmafs_FHD_fake = [max(results_fake[img]["vmaf"]["vmaf_v0.6.1"].values()) for img in results_fake]
    vmafs_4K_fake = [max(results_fake[img]["vmaf"]["vmaf_4k_v0.6.1"].values()) for img in results_fake]

    # get key of max values
    # max(psnrs[0], key=psnrs[0].get)

    plt.figure(figsize=(10,6))
    plt.hist(psnrs_fake, bins=100, color="red", alpha=0.5, label="psnr_fake")
    plt.hist(psnrs_real, bins=100, color="blue", alpha=0.5, label="psnr_real")
    plt.grid(axis="y", alpha=0.75)
    plt.xlabel("psnr value")
    plt.ylabel("frequency")
    plt.title("histgram of psnr distribution")
    plt.legend()
    plt.savefig("results/psnr.pdf", format="pdf")

    plt.figure(figsize=(10,6))
    plt.hist(ssims_fake, bins=100, color="red", alpha=0.5, label="ssim_fake")
    plt.hist(ssims_real, bins=100, color="blue", alpha=0.5, label="ssim_real")
    plt.grid(axis="y", alpha=0.75)
    plt.xlabel("ssim value")
    plt.ylabel("frequency")
    plt.title("histgram of ssim distribution")
    plt.legend()
    plt.savefig("results/ssim.pdf", format="pdf")

    plt.figure(figsize=(10,6))
    plt.hist(vmafs_FHD_fake, bins=100, color="red", alpha=0.5, label="vmaf_FHD_fake")
    plt.hist(vmafs_FHD_real, bins=100, color="blue", alpha=0.5, label="vmaf_FHD_real")
    plt.grid(axis="y", alpha=0.75)
    plt.xlabel("vmaf FHD value")
    plt.ylabel("frequency")
    plt.title("histgram of vmaf FHD distribution")
    plt.legend()
    plt.savefig("results/vmaf_FHD.pdf", format="pdf")

    plt.figure(figsize=(10,6))
    plt.hist(vmafs_4K_fake, bins=100, color="red", alpha=0.5, label="vmaf_4K_fake")
    plt.hist(vmafs_4K_real, bins=100, color="blue", alpha=0.5, label="vmaf_4K_real")
    plt.grid(axis="y", alpha=0.75)
    plt.xlabel("vmaf 4K value")
    plt.ylabel("frequency")
    plt.title("histgram of vmaf 4K distribution")
    plt.legend()
    plt.savefig("results/vmaf_4K.pdf", format="pdf")