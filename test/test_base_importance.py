import matplotlib.pyplot as plt
import numpy as np

from hidimstat.base_variable_importance import BaseVariableImportance


def test_plot_importance_axis():
    """Test argument axis of plot function"""
    n_features = 10
    vi = BaseVariableImportance()
    # Make the plot independent of data / randomness to test only the plotting function
    vi.importances_ = np.arange(n_features)
    ax_1 = vi.plot_importance(ax=None)
    assert isinstance(ax_1, plt.Axes)

    _, ax_2 = plt.subplots()
    vi.importances_ = np.random.standard_normal((3, n_features))
    ax_2_bis = vi.plot_importance(ax=ax_2)
    assert isinstance(ax_2_bis, plt.Axes)
    assert ax_2_bis == ax_2


def test_plot_importance_ascending():
    """Test argument ascending of plot function"""
    n_features = 10
    vi = BaseVariableImportance()

    # Make the plot independent of data / randomness to test only the plotting function
    vi.importances_ = np.arange(n_features)
    np.random.shuffle(vi.importances_)

    ax_decending = vi.plot_importance(ascending=False)
    assert np.all(
        ax_decending.containers[0].datavalues == np.flip(np.sort(vi.importances_))
    )

    ax_ascending = vi.plot_importance(ascending=True)
    assert np.all(ax_ascending.containers[0].datavalues == np.sort(vi.importances_))


def test_plot_importance_feature_names():
    """Test argument feature of plot function"""
    n_features = 10
    vi = BaseVariableImportance()

    # Make the plot independent of data / randomness to test only the plotting function
    vi.importances_ = np.arange(n_features)
    np.random.shuffle(vi.importances_)

    features_name = [str(j) for j in np.flip(np.argsort(vi.importances_))]
    ax_none = vi.plot_importance(feature_names=None)
    assert np.all(
        np.array([label.get_text() for label in ax_none.get_yticklabels()])
        == features_name
    )

    features_name = ["features_" + str(j) for j in np.flip(np.sort(vi.importances_))]
    ax_setup = vi.plot_importance(feature_names=features_name)
    assert np.all(
        np.array([label.get_text() for label in ax_setup.get_yticklabels()])
        == np.flip(np.array(features_name)[np.argsort(vi.importances_)])
    )

    vi.features_groups = {str(j * 2): [] for j in np.flip(np.sort(vi.importances_))}
    features_name = [str(j * 2) for j in np.flip(np.sort(vi.importances_))]
    ax_none_group = vi.plot_importance(feature_names=None)
    assert np.all(
        np.array([label.get_text() for label in ax_none_group.get_yticklabels()])
        == np.flip(np.array(features_name)[np.argsort(vi.importances_)])
    )
