import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy import stats

def plot_dataset1D(X, y, beta, title="Toy dataset"):
    """
    Plot a 1D toy dataset with the true regression line.
    Parameters:
    - X: Input features (n_samples, n_features)
    - y: Vectors of the regression  (n_samples,)
    - beta: Coefficients of the true variaable of importance (n_features,)
    - title: Title of the plot (str)
    """
    # Create a figure and a set of subplots
    fig, ([ax11, ax12], [ax21, ax22]) = plt.subplots(
        2, 2, width_ratios=[0.9, 0.01], height_ratios=[0.9, 0.1], figsize=(6.4, 4.8)
    )
    im_X = ax11.imshow(X, aspect="auto", interpolation="nearest")
    cbaxes_X = fig.add_axes([0.05, 0.275, 0.03, 0.6])
    col_X = plt.colorbar(im_X, cax=cbaxes_X, location="left")
    col_X.ax.set_xlabel("X values", loc="center", labelpad=10)
    col_X.ax.xaxis.set_label_position("top")
    ax11.set_ylabel("n samples")
    ax11.set_xlabel("n features")
    ax11.set_title("X:data", fontdict={"fontweight": "bold"})
    im_Y = ax12.imshow(np.expand_dims(y, 1), aspect="auto", interpolation="nearest")
    ax12.set_ylabel("y:regression        ", fontdict={"fontweight": "bold"})
    ax12.yaxis.tick_right()
    ax12.set_xticks([])
    cbaxes_Y = fig.add_axes([0.95, 0.275, 0.03, 0.6])
    col_Y = plt.colorbar(im_Y, cax=cbaxes_Y, location="left")
    col_Y.ax.set_xlabel("Y values   ", loc="center", labelpad=10)
    col_Y.ax.xaxis.set_label_position("top")
    ax21.imshow(np.expand_dims(beta, 0), aspect="auto", interpolation="nearest")
    ax21.set_xlabel("beta:variable of importance", fontdict={"fontweight": "bold"})
    ax21.set_yticks([])
    ax22.axis("off")
    plt.suptitle(title)
    plt.subplots_adjust(hspace=0.3, left=0.15, right=0.85)


def plot_validate_variable_importance(beta, beta_hat, vmin=0., vmax=1.):
    """
    Validate the variable importance estimation.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.4, 2.8))
    ax1.imshow(np.expand_dims(beta, 0), vmin=vmin, vmax=vmax)
    ax1.set_xlabel("beta:variable of importance", fontdict={"fontweight": "bold"})
    ax1.set_yticks([])
    im = ax2.imshow(np.expand_dims(beta_hat, 0), vmin=vmin, vmax=vmax)
    ax2.set_xlabel("beta hat:variable of importance", fontdict={"fontweight": "bold"})
    ax2.set_yticks([])
    plt.colorbar(
        im, ax=ax2, orientation="horizontal", label="Variable importance", pad=0.5
    )
    plt.subplots_adjust(top=1.0, bottom=0.2)


def plot_pvalue_H0(beta_hat, pvalue, pvalue_corrected, one_minus_pvalue, one_minus_pvalue_corrected, vmin=0., vmax=1.):
    """
    Validate the variable is importance.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6.4, 4.8))
    im_beta_hat = ax1.imshow(np.expand_dims(beta_hat, 0), vmin=vmin, vmax=vmax)
    ax1.set_title("beta hat:variable of importance", fontdict={"fontweight": "bold"})
    ax1.set_yticks([])
    colbar_beta_hat = plt.colorbar(
        im_beta_hat, ax=ax1, orientation="horizontal", label="Variable importance", pad=0.2
    )
    colbar_beta_hat.ax.xaxis.labelpad = 0
    im_pvalue = ax2.imshow(np.expand_dims(pvalue, 0), norm=LogNorm(vmin=np.min(pvalue), vmax=np.max(pvalue)), cmap=plt.cm.viridis.reversed())
    ax2.set_title("pvalue of each variable of importance", fontdict={"fontweight": "bold"})
    ax2.set_yticks([])
    colbar_pvalue = plt.colorbar(
        im_pvalue, ax=ax2, orientation="horizontal", label="pvalue", pad=0.2
    )
    colbar_pvalue.ax.xaxis.labelpad = 0
    im_pvalue_corrected = ax3.imshow(np.expand_dims(pvalue_corrected, 0), norm=LogNorm(vmin=np.min(pvalue_corrected), vmax=np.max(pvalue_corrected)), cmap=plt.cm.viridis.reversed())
    ax3.set_title("corrected pvalue of each variable of importance", fontdict={"fontweight": "bold"})
    ax3.set_yticks([])
    colbar_pvalue_corrected = plt.colorbar(
        im_pvalue_corrected, ax=ax3, orientation="horizontal", label="corrected pvalue", pad=0.2
    )
    colbar_pvalue_corrected.ax.xaxis.labelpad = 0
    plt.subplots_adjust(top=1.0, hspace=0.3)
    
def plot_pvalue_H1(beta_hat, pvalue, pvalue_corrected, one_minus_pvalue, one_minus_pvalue_corrected, vmin=0., vmax=1.):
    """
    Validate the variable is not importance.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6.4, 4.8))
    im_beta_hat = ax1.imshow(np.expand_dims(beta_hat, 0), vmin=vmin, vmax=vmax)
    ax1.set_title("beta hat:variable of importance", fontdict={"fontweight": "bold"})
    ax1.set_yticks([])
    colbar_beta_hat = plt.colorbar(
        im_beta_hat, ax=ax1, orientation="horizontal", label="Variable importance", pad=0.2
    )
    colbar_beta_hat.ax.xaxis.labelpad = 0
    im_pvalue = ax2.imshow(np.expand_dims(one_minus_pvalue_corrected, 0), norm=LogNorm(vmin=np.min(one_minus_pvalue), vmax=np.max(one_minus_pvalue_corrected)), cmap=plt.cm.viridis.reversed())
    ax2.set_title("pvalue of each variable of importance", fontdict={"fontweight": "bold"})
    ax2.set_yticks([])
    colbar_pvalue = plt.colorbar(
        im_pvalue, ax=ax2, orientation="horizontal", label="pvalue", pad=0.2
    )
    colbar_pvalue.ax.xaxis.labelpad = 0
    im_pvalue_corrected = ax3.imshow(np.expand_dims(one_minus_pvalue_corrected, 0), norm=LogNorm(vmin=np.min(one_minus_pvalue_corrected), vmax=np.max(one_minus_pvalue_corrected)), cmap=plt.cm.viridis.reversed())
    ax3.set_title("corrected pvalue of each variable of importance", fontdict={"fontweight": "bold"})
    ax3.set_yticks([])
    colbar_pvalue_corrected = plt.colorbar(
        im_pvalue_corrected, ax=ax3, orientation="horizontal", label="corrected pvalue", pad=0.2
    )
    colbar_pvalue_corrected.ax.xaxis.labelpad = 0
    plt.subplots_adjust(top=1.0, hspace=0.3)

def plot_compare_proba_estimated(proba, beta_hat, scale):
    fig, axs = plt.subplots(5, 5, figsize=(6.4, 4.8))
    for index, ax in enumerate(axs.flat):
        ax.hist(proba[:,index], bins=100, density=True, alpha=0.5, color="b", label="proba")
        x = np.linspace( - 3*scale[index], + 3*scale[index], 100)
        ax.plot(x, stats.norm.pdf(x, 0.0, scale[index]), 'r-', lw=2, label="norm pdf")
        ax.set_title(f"beta_hat[{index}]")