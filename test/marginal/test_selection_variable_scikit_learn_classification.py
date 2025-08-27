from copy import deepcopy
import numpy as np
import pytest

from hidimstat import ANOVA, MutualInformation, AdapterScikitLearn


class MutualInformationClassification(MutualInformation):
    """Specify the class for classification problem"""

    def __init__(
        self,
        discrete_features="auto",
        n_neighbors=3,
        random_state=None,
        n_jobs=1,
    ):
        super().__init__(
            problem_type="classification",
            discrete_features=discrete_features,
            n_neighbors=n_neighbors,
            random_state=random_state,
            n_jobs=n_jobs,
        )


def classification_float(y, nb_classes=4, min_value=None, max_value=None):
    """
    Create classification problem bae on regression problem
    Parameters
    ----------
    y : array-like of shape (n_samples,)
        target for the prediction
    nb_classes : int, default=4
        number of classes
    min_value : None or int, default=None
        maximal value for the first class
    max_value : None or int, default=None
        minimal value for the last class
    Returns
    -------
    array-like of shape (n_samples,)
        classification for target
    """
    assert nb_classes >= 2
    if min_value is None:
        min_value = np.min(y)
    if max_value is None:
        max_value = np.max(y)
    assert min_value < max_value
    # change from regression to classification problem
    y_ = deepcopy(y)
    previous_value = min_value
    for classe, value in enumerate(np.linspace(min_value, max_value, nb_classes)):
        if value == min_value:
            y_[np.where(y < min_value)] = classe
        elif value == max_value:
            y_[np.where(y >= max_value)] = classe
        else:
            y_[np.where(np.logical_and(previous_value <= y, y < value))] = classe
        previous_value = value
    y_ = np.array(y_, dtype=int)
    return y_


def configure_marginal_classication(
    ClassMethod, X, y, nb_classes=4, min_value=None, max_value=None
):
    """
    Configure ClassMethod model for feature importance analysis.
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data matrix where each column represents a feature
        and each row a sample.
    y : array-like of shape (n_samples,)
        Target variable array.
    Returns
    -------
    importance : array-like
        Array containing importance scores for each feature.
        Higher values indicate greater feature importance in predicting
        the target variable.
    Notes
    -----
    The function performs the following steps:
    1. Intanciate ClassMethod
    2. Calculates feature importance
    """
    y_ = classification_float(y, nb_classes, min_value, max_value)
    # instantiate model
    vi = ClassMethod()
    # fit the model using the training set
    vi.fit()
    # calculate feature importance using the test set
    importance = vi.importance(X, y_)
    return np.array(importance)


parameter_exact = [
    ("HiDim", 150, 200, 1, 0.0, 42, 1.0, np.inf, 0.0),
    ("HiDim with noise", 150, 200, 1, 0.0, 42, 1.0, 10.0, 0.0),
    ("HiDim with correlated noise", 150, 200, 1, 0.0, 42, 1.0, 10.0, 0.5),
    ("HiDim with correlated features", 150, 200, 1, 0.8, 42, 1.0, np.inf, 0.0),
]


@pytest.mark.parametrize(
    "n_samples, n_features, support_size, rho, seed, value, signal_noise_ratio, rho_serial",
    zip(*(list(zip(*parameter_exact))[1:])),
    ids=list(zip(*parameter_exact))[0],
)
@pytest.mark.parametrize(
    "ClassVI",
    [ANOVA, MutualInformationClassification],
    ids=["ANOVA", "MutualInformation"],
)
def test_linear_data_exact(data_generator, ClassVI):
    """Tests the method on linear cases with noise and/or correlation"""
    X, y, important_features, _ = data_generator

    importance = configure_marginal_classication(ClassVI, X, y)
    # check that importance scores are defined for each feature
    assert importance.shape == (X.shape[1],)
    # check that important features have the highest importance scores
    assert np.all([int(i) in important_features for i in np.argsort(importance)[-1:]])


parameter_bad_detection = [
    ("HiDim with high correlated features", 150, 200, 1, 1.0, 42, 1.0, 5.0, 0.0),
    ("HiDim multivaribale", 150, 200, 10, 0.0, 42, 1.0, np.inf, 0.0),
    ("HiDim multivaribale noise", 150, 200, 10, 0.0, 42, 1.0, 10.0, 0.0),
]


@pytest.mark.parametrize(
    "n_samples, n_features, support_size, rho, seed, value, signal_noise_ratio, rho_serial",
    zip(*(list(zip(*parameter_bad_detection))[1:])),
    ids=list(zip(*parameter_bad_detection))[0],
)
@pytest.mark.parametrize(
    "ClassVI",
    [ANOVA, MutualInformationClassification],
    ids=["ANOVA", "MutualInformation"],
)
def test_linear_data_fail(data_generator, ClassVI):
    """Tests the faillure of the method on linear cases with correlation
    or multiple variable of importance"""
    X, y, important_features, _ = data_generator
    size_support = np.sum(important_features != 0)

    importance = configure_marginal_classication(ClassVI, X, y)
    # check that importance scores are defined for each feature
    assert importance.shape == (X.shape[1],)
    # check that important features have the highest importance scores
    assert np.any(
        [
            int(i) not in important_features
            for i in np.argsort(importance)[-size_support:]
        ]
    )


################################################################################
# Specific test for ANOVA
parameter_exact_ANOVA = [
    ("HiDim with high level noise", 150, 200, 1, 0.2, 42, 1.0, 0.5, 0.0),
]


@pytest.mark.parametrize(
    "n_samples, n_features, support_size, rho, seed, value, signal_noise_ratio, rho_serial",
    zip(*(list(zip(*parameter_exact_ANOVA))[1:])),
    ids=list(zip(*parameter_exact_ANOVA))[0],
)
def test_ANOVA_exact(data_generator):
    """Tests the method on high noise"""
    X, y, important_features, not_important_features = data_generator

    importance = configure_marginal_classication(ANOVA, X, y)
    # check that importance scores are defined for each feature
    assert importance.shape == (X.shape[1],)
    # check that important features have the highest importance scores
    assert np.all([int(i) in important_features for i in np.argsort(importance)[-1:]])
    # Check that important features have higher mean importance scores
    assert (
        importance[important_features].mean()
        > importance[not_important_features][
            np.where(importance[not_important_features] != 0)
        ].mean()
    )


# Spefic test for MutualInformation
@pytest.mark.parametrize(
    "n_samples, n_features, support_size, rho, seed, value, signal_noise_ratio, rho_serial",
    [parameter_exact[0][1:]],
    ids=[parameter_exact[0][0]],
)
@pytest.mark.parametrize(
    "discrete_features, n_neighbors",
    [
        ("auto", 5),
        (False, 5),
    ],
    ids=[
        "change number of neighboor",
        "discrete_features True",
    ],
)
def test_MutualInformation_exact(data_generator, discrete_features, n_neighbors):
    """Tests parameters of classes"""
    X, y, important_features, _ = data_generator
    y_ = classification_float(y, nb_classes=6, min_value=-1, max_value=1)

    importance = (
        MutualInformationClassification(
            discrete_features=discrete_features, n_neighbors=n_neighbors
        )
        .fit()
        .importance(X, y_)
    )
    # check that importance scores are defined for each feature
    assert importance.shape == (X.shape[1],)
    # check that important features have the highest importance scores
    assert np.all([int(i) in important_features for i in np.argsort(importance)[-1:]])


parameter_fail_MutualInformation = [
    ("HiDim with high level noise", 150, 200, 1, 0.2, 42, 1.0, 0.5, 0.0),
]


@pytest.mark.parametrize(
    "n_samples, n_features, support_size, rho, seed, value, signal_noise_ratio, rho_serial",
    zip(*(list(zip(*parameter_fail_MutualInformation))[1:])),
    ids=list(zip(*parameter_fail_MutualInformation))[0],
)
def test_MutualInformation_fail(data_generator):
    """Tests faillure of the method on high noise"""
    X, y, important_features, _ = data_generator
    size_support = np.sum(important_features != 0)

    importance = configure_marginal_classication(MutualInformationClassification, X, y)
    # check that importance scores are defined for each feature
    assert importance.shape == (X.shape[1],)
    # check that important features have the highest importance scores
    assert np.any(
        [
            int(i) not in important_features
            for i in np.argsort(importance)[-size_support:]
        ]
    )


##############################################################################
@pytest.mark.parametrize(
    "n_samples, n_features, support_size, rho, seed, value, signal_noise_ratio, rho_serial",
    [(150, 200, 1, 0.0, 42, 1.0, 0.0, 0.0)],
    ids=["default data"],
)
@pytest.mark.parametrize(
    "ClassVI",
    [ANOVA, MutualInformationClassification],
    ids=["ANOVA", "MutualInformation"],
)
class TestClass:
    """Test the element of the class"""

    def test_init(self, data_generator, ClassVI):
        """Test initialization work"""
        classvi = ClassVI()

    def test_fit(self, data_generator, ClassVI):
        """Test fitting is doing nothing"""
        classvi = ClassVI()
        classvi_reference = deepcopy(classvi)
        classvi.fit()
        for attribute_name in classvi.__dict__.keys():
            assert classvi.__getattribute__(
                attribute_name
            ) == classvi_reference.__getattribute__(attribute_name)

    def test_categorical(
        self,
        n_samples,
        n_features,
        support_size,
        rho,
        seed,
        value,
        signal_noise_ratio,
        rho_serial,
        ClassVI,
    ):
        """Test the fit_importance function on mix type of feature"""
        rng = np.random.default_rng(seed)
        X_cont = rng.random((n_samples, 2))
        X_cat = rng.integers(low=0, high=3, size=(n_samples, 1))
        X = np.hstack([X_cont, X_cat])
        y = rng.integers(0, 10, (n_samples, 1))

        classvi = ClassVI()

        importances = classvi.fit_importance(X, y)
        assert len(importances) == 3
        assert np.all(importances >= 0)


##############################################################################
def test_error_abstract_class():
    """Test the warning and the error of the class AdapterScikitLearn"""
    adapter = AdapterScikitLearn()
    with pytest.warns(Warning, match="X won't be used"):
        adapter.fit(X=np.random.rand(10, 10))
    with pytest.warns(Warning, match="y won't be used"):
        adapter.fit(y=np.random.rand(10, 1))
    with pytest.raises(NotImplementedError):
        adapter.importance(X=np.random.rand(10, 10), y=np.random.rand(10, 1))
    with pytest.raises(NotImplementedError):
        adapter.fit_importance(X=np.random.rand(10, 10), y=np.random.rand(10, 1))
    with pytest.raises(NotImplementedError):
        with pytest.warns(Warning, match="cv won't be used"):
            adapter.fit_importance(
                X=np.random.rand(10, 10), y=np.random.rand(10, 1), cv="other"
            )


def test_error_Mutual_Information():
    """Test error in Mutual Information"""
    with pytest.raises(
        AssertionError,
        match="the value of problem type should be 'regression' or 'classification'",
    ):
        MutualInformation(problem_type="bad type")
    vi = MutualInformation()
    vi.problem_type = "bad type"
    with pytest.raises(
        ValueError,
        match="the value of problem type should be 'regression' or 'classification'",
    ):
        vi.importance(X=np.random.rand(10, 10), y=np.random.rand(10, 1))
