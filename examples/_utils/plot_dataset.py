import numpy as np
import matplotlib.pyplot as plt

def plot_dataset1D(X, y, beta, title='Toy dataset'):
    """
    Plot a 1D toy dataset with the true regression line.
    Parameters:
    - X: Input features (n_samples, n_features)
    - y: Vectors of the regression  (n_samples,)
    - beta: Coefficients of the true variaable of importance (n_features,)
    - title: Title of the plot (str)
    """
    # Create a figure and a set of subplots
    fig, ([ax11, ax12], [ax21, ax22]) = plt.subplots(2, 2, width_ratios=[0.9, 0.01], height_ratios=[0.9, 0.1])
    ax11.imshow(X, aspect='auto', interpolation='nearest')
    ax11.set_ylabel('n samples')
    ax11.set_xlabel('n features')
    ax11.set_title('X:data', fontdict={'fontweight': 'bold'})
    ax12.imshow(np.expand_dims(y,1), aspect='auto', interpolation='nearest')
    ax12.set_title('y:regression', fontdict={'fontweight': 'bold'})
    ax12.yaxis.tick_right()
    ax12.set_xticks([])
    ax21.imshow(np.expand_dims(beta,0), aspect='auto', interpolation='nearest')
    ax21.set_xlabel('beta:variable of importance', fontdict={'fontweight': 'bold'})
    ax21.set_yticks([])
    ax22.axis('off')
    plt.suptitle(title)
    plt.subplots_adjust(hspace=0.3)
    fig.tight_layout()
    # plt.show()
