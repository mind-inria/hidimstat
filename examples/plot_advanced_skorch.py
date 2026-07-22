"""
Using PyTorch with HiDimStat
============================
HiDimStat was designed with the goal of being easily compatible with Sklearn.
PyTorch models might not be used directly with HiDimStat for that reason.
However, with the help of a third party library `Skorch <https://skorch.readthedocs.io/en/stable/>`_,
PyTorch can be interfaced with HiDimStat, and provide all of its functionalities.
In this example, we define a Convolutional Neural Network (CNN) in Skorch,
and perform pixel-wise feature importance in a binary classification setup,
through the MNIST digits dataset, between digits 4 vs 7 and 0 vs 1.
"""

# %%
# Defining a PyTorch model
# -------------------------
# We start by defining a basic Convolutional Neural Network (CNN)
# composed of 2 series of (Convolution, Activation, Pooling) layers,
# followed by 2 fully-connected layers.

import torch.nn.functional as F
from torch import nn


class MNISTCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2)

        # 28x28 -> 14x14 -> 7x7
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # We input an array of shape (batch_size, n_features=784)
        # but conv1 expects a tensor of shape (batch_size, n_channels=1, height=28, width=28)
        x = x.view((-1, 1, 28, 28)).float()
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.flatten(1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


# %%
# Loading data
# ------------
# Since Skorch interfaces PyTorch in a Scikit-learn fashion, we don't need to create torch datasets, and we simply prepare
# data as we would usually do for sklearn models.

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.utils import resample

mnist_dataset = fetch_openml("mnist_784", version=1, as_frame=False)
X_mnist, y_mnist = mnist_dataset.data, mnist_dataset.target

# Downsample to speed up the example
n_samples = 500
mask_4_7 = (y_mnist == "4") | (y_mnist == "7")
X_4_7, y_4_7 = (
    X_mnist[mask_4_7].astype(np.float32),
    y_mnist[mask_4_7].astype(int),
)
X_4_7, y_4_7 = resample(
    X_4_7,
    y_4_7,
    n_samples=n_samples,
    replace=False,
    random_state=0,
    stratify=y_4_7,
)

# Plot digits 0 against 1, and 4 against 7, to visually compare a few of them.
_, axes = plt.subplots(
    1, 5, figsize=(4, 2), subplot_kw={"xticks": [], "yticks": []}
)
for i in range(5):
    # Plot 4 vs 7
    label = 7 if i % 2 == 0 else 4
    axes[i].imshow(X_4_7[y_4_7 == label][i].reshape(28, 28), cmap="gray")

axes[2].set_title("Digits 4 vs 7", fontweight="bold", y=1.0)


# %%
# We now create the Skorch model, and build a pipeline with a StandardScaler.
# Skorch allows the model to be run either on a cuda device or on cpu.

import torch.cuda
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from skorch import NeuralNetClassifier
from torch.optim import Adam

net = NeuralNetClassifier(
    MNISTCNN,
    max_epochs=20,
    lr=1e-3,
    batch_size=64,
    optimizer=Adam,
    criterion=nn.CrossEntropyLoss,
    iterator_train__shuffle=True,
    device="cuda" if torch.cuda.is_available() else "cpu",
)

# Current hotfix of API compatibility issue (Skorch doesn't automatically set this during fitting).
# This needs to be set, otherwise HiDimStat throws an error.
net.n_features_in_ = 28 * 28

# %%
# Running HiDimStat feature importance computation
# ------------------------------------------------
# We cluster pixels through feature agglomeration, while leveraging their grid structure
# and assess feature importance at the group level.
# This is done with a Conditional Feature Importance (CFI). For each binary
# classification (0 vs 1, 4 vs 7), we fit the model and evaluate feature
# importance. We also plot the clusters formed by the FeatureAgglomeration
# to visualize the "superpixels" formed this way.

from joblib import parallel_config
from sklearn.cluster import FeatureAgglomeration
from sklearn.feature_extraction import image
from sklearn.linear_model import LinearRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

from hidimstat import CFI

shape = (28, 28)
n_clusters = 50
target_fwer = 0.1

X_cluster = resample(
    X_4_7, n_samples=n_samples // 2, replace=False, random_state=0
)
connectivity = image.grid_to_graph(n_x=shape[0], n_y=shape[1])
clustering = FeatureAgglomeration(
    n_clusters=n_clusters, connectivity=connectivity
)
clustering.fit(X_cluster)

# CFI expects features_groups to be a dictionary where the keys are the group names
# and the values are the list of column names corresponding to each features group.
order = np.argsort(clustering.labels_)
sorted_ids = clustering.labels_[order]
unique_ids, start_idx = np.unique(sorted_ids, return_index=True)
positions = np.split(order, start_idx[1:])
features_groups = dict(zip(unique_ids, positions, strict=False))

# %%
# We now instantiate the CFI and fit it on the two tasks.

# Careful when using Skorch, having n_jobs > 1 might create joblib and pickle issues.
# This can be avoided as follows:
with parallel_config(backend="threading"):
    cfi = CFI(
        estimator=net,
        features_groups=features_groups,
        imputation_model_continuous=LinearRegression(),
        n_permutations=5,
        method="predict_proba",
        loss=log_loss,
        random_state=0,
        n_jobs=-1,
    )
    # PyTorch expects float input, and long type target.
    # Since it's a binary classification, we convert the classes into 4->0 and 7->1 for Skorch.
    y_target = y_4_7.astype(np.int64).copy()
    y_target[y_target == 4] = 0
    y_target[y_target == 7] = 1

    X_train, X_test, y_train, y_test = train_test_split(
        X_4_7, y_target, test_size=0.3, random_state=0, stratify=y_target
    )
    net.fit(X_train, y_train)
    print(f"Model accuracy on the test set: {net.score(X_test, y_test):.2f}")
    cfi.fit(X_train, y_train)
    importances_4_7 = cfi.importance(X_test, y_test)
    selected_4_7 = cfi.fwer_selection(
        fwer=target_fwer, n_tests=n_clusters, two_tailed_test=True
    )

# %%
# Visualizing the results
# -----------------------
# Finally, we visualize the significant pixels identified by CFI for each of the
# classification tasks.

from matplotlib.colors import ListedColormap

fig, ax = plt.subplots(
    1, 3, figsize=(5, 2), subplot_kw={"xticks": [], "yticks": []}
)

pixel_selection = clustering.inverse_transform(selected_4_7)
mask_cfi = pixel_selection.reshape(shape)
importance_cfi_img = clustering.inverse_transform(importances_4_7).reshape(
    shape
)

cmap = ListedColormap(["tab:red", "white", "tab:blue"])
ax[0].imshow(mask_cfi, cmap=cmap, vmin=-1, vmax=1)
ax[0].set_title("4 vs 7 selection", fontweight="bold", y=1.0)

im = ax[1].imshow(importance_cfi_img, cmap="bone")
ax[1].set_title("4 vs 7 importance", fontweight="bold", y=1.0)
cbar = fig.colorbar(im, ax=ax[1], fraction=0.044)
cbar.set_label("Importance")

ax[2].imshow(clustering.labels_.reshape(28, 28), cmap="Set2")
ax[2].axis("off")
ax[2].set_title("Clustering labels", fontweight="bold", y=1.0)

plt.tight_layout()
plt.show()
