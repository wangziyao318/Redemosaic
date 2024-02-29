# Code source: Gaël Varoquaux
#              Andreas Müller
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from sklearn.datasets import make_circles, make_classification, make_moons
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

names = [
    "Nearest Neighbors",
    "Linear SVM",
    "RBF SVM",
    "Gaussian Process",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
    "Naive Bayes",
    "QDA",
]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025, random_state=42),
    SVC(gamma=2, C=1, random_state=42),
    GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42),
    DecisionTreeClassifier(max_depth=5, random_state=42),
    RandomForestClassifier(
        max_depth=5, n_estimators=10, max_features=1, random_state=42
    ),
    MLPClassifier(alpha=1, max_iter=1000, random_state=42),
    AdaBoostClassifier(algorithm="SAMME", random_state=42),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
]


try:
    with open("results_real.json", "r") as f:
        results_real = json.load(f)
    with open("results_fake.json", "r") as f:
        results_fake = json.load(f)
except OSError:
    raise
except json.decoder.JSONDecodeError:
    raise

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

psnr = psnr_real + psnr_fake
ssim = ssim_real + ssim_fake
vmaf = vmaf_FHD_real + vmaf_FHD_fake

y = [True]*len(psnr_real) + [False]*len(psnr_fake)
y = np.array(y)

datasets = []

X = [psnr, ssim]
X = [list(i) for i in zip(*X)]
X = np.array(X)
datasets.append((X,y))

X = [psnr, vmaf]
X = [list(i) for i in zip(*X)]
X = np.array(X)
datasets.append((X,y))

X = [ssim, vmaf]
X = [list(i) for i in zip(*X)]
X = np.array(X)
datasets.append((X,y))


# psnr_T = np.array(psnr_real, dtype=np.float64)
# psnr_F = np.array(psnr_fake, dtype=np.float64)
# X_psnr = np.concatenate([psnr_T, psnr_F], axis=0)

# ssim_T = np.array(ssim_real, dtype=np.float64)
# ssim_F = np.array(ssim_fake, dtype=np.float64)
# X_ssim = np.concatenate([ssim_T, ssim_F], axis=0)

# vmaf_FHD_T = np.array(vmaf_FHD_real, dtype=np.float64)
# vmaf_FHD_F = np.array(vmaf_FHD_fake, dtype=np.float64)
# X_vmaf_FHD = np.concatenate([vmaf_FHD_T, vmaf_FHD_F], axis=0)

# vmaf_4K_T = np.array(vmaf_4K_real, dtype=np.float64)
# vmaf_4K_F = np.array(vmaf_4K_fake, dtype=np.float64)
# X_vmaf_4K = np.concatenate([vmaf_4K_T, vmaf_4K_F], axis=0)

# X = np.stack([X_psnr, X_ssim], axis=1)

# y_T = np.ones_like(psnr_T, dtype=np.bool_)
# y_F = np.zeros_like(psnr_F, dtype=np.bool_)
# y = np.concatenate([y_T, y_F], axis=0)


# X, y = make_classification(
#     n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1
# )
# rng = np.random.RandomState(2)
# X += 2 * rng.uniform(size=X.shape)
# linearly_separable = (X, y)

figure = plt.figure(figsize=(27, 9))
i = 1
# iterate over datasets
for ds_cnt, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
    X, y = ds
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42
    )

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(["#FF0000", "#0000FF"])
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    if ds_cnt == 0:
        ax.set_title("Input data")
    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k")
    # Plot the testing points
    ax.scatter(
        X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors="k"
    )
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)

        clf = make_pipeline(StandardScaler(), clf)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        DecisionBoundaryDisplay.from_estimator(
            clf, X, cmap=cm, alpha=0.8, ax=ax, eps=0.5
        )

        # Plot the training points
        ax.scatter(
            X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k"
        )
        # Plot the testing points
        ax.scatter(
            X_test[:, 0],
            X_test[:, 1],
            c=y_test,
            cmap=cm_bright,
            edgecolors="k",
            alpha=0.6,
        )

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(())
        ax.set_yticks(())
        if ds_cnt == 0:
            ax.set_title(name)
        ax.text(
            x_max - 0.3,
            y_min + 0.3,
            ("%.2f" % score).lstrip("0"),
            size=15,
            horizontalalignment="right",
        )
        i += 1

plt.tight_layout()
plt.savefig("results/sample_classifier.pdf", format="pdf")
plt.show()