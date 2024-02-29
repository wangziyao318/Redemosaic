import json
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import numpy as np

from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    try:
        with open("results_real.json", "r") as f:
            results_real = json.load(f)
        with open("results_fake.json", "r") as f:
            results_fake = json.load(f)
    except OSError:
        raise
    except json.decoder.JSONDecodeError:
        raise
    if (results_real == {}) or (results_fake == {}):
        print("empty results real/fake")
        exit
    
    psnr_real = [max(results_real[img]["psnr"].values()) for img in results_real]
    ssim_real = [max(results_real[img]["ssim"].values()) for img in results_real]
    vmaf_FHD_real = [max(results_real[img]["vmaf"]["vmaf_v0.6.1"].values()) for img in results_real]
    # vmaf_4K_real = [max(results_real[img]["vmaf"]["vmaf_4k_v0.6.1"].values()) for img in results_real]

    psnr_fake = [max(results_fake[img]["psnr"].values()) for img in results_fake]
    ssim_fake = [max(results_fake[img]["ssim"].values()) for img in results_fake]
    vmaf_FHD_fake = [max(results_fake[img]["vmaf"]["vmaf_v0.6.1"].values()) for img in results_fake]
    # vmaf_4K_fake = [max(results_fake[img]["vmaf"]["vmaf_4k_v0.6.1"].values()) for img in results_fake]

    # get key of max values
    # max(psnrs[0], key=psnrs[0].get)

    psnr_T = np.array(psnr_real, dtype=np.float64)
    psnr_F = np.array(psnr_fake, dtype=np.float64)
    X_psnr = np.concatenate([psnr_T, psnr_F], axis=0)

    ssim_T = np.array(ssim_real, dtype=np.float64)
    ssim_F = np.array(ssim_fake, dtype=np.float64)
    X_ssim = np.concatenate([ssim_T, ssim_F], axis=0)

    vmaf_FHD_T = np.array(vmaf_FHD_real, dtype=np.float64)
    vmaf_FHD_F = np.array(vmaf_FHD_fake, dtype=np.float64)
    X_vmaf_FHD = np.concatenate([vmaf_FHD_T, vmaf_FHD_F], axis=0)

    # vmaf_4K_T = np.array(vmaf_4K_real, dtype=np.float64)
    # vmaf_4K_F = np.array(vmaf_4K_fake, dtype=np.float64)
    # X_vmaf_4K = np.concatenate([vmaf_4K_T, vmaf_4K_F], axis=0)

    # n examples by d features
    X = np.stack([X_psnr, X_ssim, X_vmaf_FHD], axis=1)

    y_T = np.ones_like(psnr_T, dtype=np.bool_)
    y_F = np.zeros_like(psnr_F, dtype=np.bool_)
    # n examples
    y = np.concatenate([y_T, y_F], axis=0)


    datasets = train_test_split(X, y, test_size=0.3)

    train_data, test_data, train_labels, test_labels = datasets

    print(f"{train_labels.shape[0]} training data, {test_labels.shape[0]} test data")

    clf = LogisticRegression(random_state=0, solver="lbfgs", multi_class="multinomial", max_iter=1000).fit(train_data, train_labels)

    print("multi-class logistic regression")
    print(clf.score(train_data, train_labels))
    print(clf.score(test_data, test_labels))
    """
    all data classifier accuracy:
    0.847423711855928 for psnr
    0.8414207103551776 for ssim
    0.8169084542271136 for vmaf_FHD
    0.7358679339669835 for vmaf_4K (discarded as bad classifier)
    0.886943471735868 for combined psnr, ssim, vmaf_FHD in multinomial
    """

    clf_MLP = MLPClassifier(hidden_layer_sizes=[100,], random_state=1, max_iter=1000)
    clf_MLP.fit(train_data,train_labels)

    print("unoptimized neural network")
    print(clf_MLP.score(train_data, train_labels))
    print(clf_MLP.score(test_data, test_labels))


    # plt.figure(figsize=(10,6))
    # plt.hist(psnr_fake, bins=100, color="red", alpha=0.5, label="psnr_fake")
    # plt.hist(psnr_real, bins=100, color="blue", alpha=0.5, label="psnr_real")
    # plt.grid(axis="y", alpha=0.75)
    # plt.xlabel("psnr value")
    # plt.ylabel("frequency")
    # plt.title("histgram of psnr distribution")
    # plt.legend()
    # plt.savefig("results/psnr.pdf", format="pdf")

    # plt.figure(figsize=(10,6))
    # plt.hist(ssim_fake, bins=100, color="red", alpha=0.5, label="ssim_fake")
    # plt.hist(ssim_real, bins=100, color="blue", alpha=0.5, label="ssim_real")
    # plt.grid(axis="y", alpha=0.75)
    # plt.xlabel("ssim value")
    # plt.ylabel("frequency")
    # plt.title("histgram of ssim distribution")
    # plt.legend()
    # plt.savefig("results/ssim.pdf", format="pdf")

    # plt.figure(figsize=(10,6))
    # plt.hist(vmaf_FHD_fake, bins=100, color="red", alpha=0.5, label="vmaf_FHD_fake")
    # plt.hist(vmaf_FHD_real, bins=100, color="blue", alpha=0.5, label="vmaf_FHD_real")
    # plt.grid(axis="y", alpha=0.75)
    # plt.xlabel("vmaf FHD value")
    # plt.ylabel("frequency")
    # plt.title("histgram of vmaf FHD distribution")
    # plt.legend()
    # plt.savefig("results/vmaf_FHD.pdf", format="pdf")

    # plt.figure(figsize=(10,6))
    # plt.hist(vmaf_4K_fake, bins=100, color="red", alpha=0.5, label="vmaf_4K_fake")
    # plt.hist(vmaf_4K_real, bins=100, color="blue", alpha=0.5, label="vmaf_4K_real")
    # plt.grid(axis="y", alpha=0.75)
    # plt.xlabel("vmaf 4K value")
    # plt.ylabel("frequency")
    # plt.title("histgram of vmaf 4K distribution")
    # plt.legend()
    # plt.savefig("results/vmaf_4K.pdf", format="pdf")

    figure = plt.figure()
    ax = figure.add_subplot(projection="3d")
    ax.scatter(psnr_real, ssim_real, vmaf_FHD_real, color="blue", marker="o", alpha=0.5, label="real")
    ax.scatter(psnr_fake, ssim_fake, vmaf_FHD_fake, color="red", marker="^", alpha=0.5, label="fake")
    ax.set_xlabel("psnr value")
    ax.set_ylabel("ssim value")
    ax.set_zlabel("vmaf value")
    ax.set_title("psnr v ssim v vmaf")
    ax.legend()
    plt.savefig("results/psnr_ssim_vmaf.pdf", format="pdf")
    plt.show()