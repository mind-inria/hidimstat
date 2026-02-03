import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.utils.estimator_checks import parametrize_with_checks

from hidimstat.base_perturbation import BasePerturbation, BasePerturbationCV

from .conftest import SKLEARN_LT_1_6, check_estimator


def _fitted_linear_regression():
    X = np.random.randint(0, 2, size=(100, 2, 1))
    estimator = LinearRegression()
    estimator.fit(X[:, 0], X[:, 1])
    return estimator


ESTIMATORS_TO_CHECK = [
    BasePerturbation(estimator=_fitted_linear_regression()),
    BasePerturbation(estimator=LinearRegression()),
    BasePerturbationCV(estimators=LinearRegression(), cv=KFold(n_splits=2)),
    # TODO
    # BasePerturbationCV(
    #     estimators=[_fitted_linear_regression(), _fitted_linear_regression()],
    #     cv=KFold(n_splits=2),
    # ),
]


def expected_failed_checks(estimator):
    if isinstance(estimator, BasePerturbation):
        return {
            "check_dont_overwrite_parameters": "TODO",
            "check_dtype_object": "TODO",
            "check_estimator_sparse_tag": "TODO",
            "check_estimators_overwrite_params": "TODO",
            "check_fit_check_is_fitted": "TODO",
            "check_n_features_in": "TODO",
            "check_n_features_in_after_fitting": "TODO",
            "check_no_attributes_set_in_init": "TODO",
        }
    elif isinstance(estimator, BasePerturbationCV):
        return {
            "check_dict_unchanged": "TODO",
            "check_do_not_raise_errors_in_init_or_set_params": "TODO",
            "check_dont_overwrite_parameters": "TODO",
            "check_dtype_object": "TODO",
            "check_estimator_cloneable0": "TODO",
            "check_estimator_cloneable1": "TODO",
            "check_estimator_sparse_array": "TODO",
            "check_estimator_sparse_matrix": "TODO",
            "check_estimator_sparse_tag": "TODO",
            "check_estimators_dtypes": "TODO",
            "check_estimators_fit_returns_self": "TODO",
            "check_estimators_overwrite_params": "TODO",
            "check_estimators_pickle": "TODO",
            "check_estimators_nan_inf": "TODO",
            "check_f_contiguous_array_estimator": "TODO",
            "check_fit_check_is_fitted": "TODO",
            "check_fit_idempotent": "TODO",
            "check_fit_score_takes_y": "TODO",
            "check_fit2d_1feature": "TODO",
            "check_fit2d_predict1d": "TODO",
            "check_parameters_default_constructible": "TODO",
            "check_n_features_in": "TODO",
            "check_n_features_in_after_fitting": "TODO",
            "check_no_attributes_set_in_init": "TODO",
            "check_methods_sample_order_invariance": "TODO",
            "check_methods_subset_invariance": "TODO",
            "check_pipeline_consistency": "TODO",
            "check_positive_only_tag_during_fit": "TODO",
            "check_readonly_memmap_input": "TODO",
        }


if SKLEARN_LT_1_6:

    @pytest.mark.parametrize(
        "estimator, check, name",
        check_estimator(
            estimators=ESTIMATORS_TO_CHECK,
            return_expected_failed_checks=expected_failed_checks,
        ),
    )
    def test_check_estimator_sklearn_valid(estimator, check, name):  # noqa: ARG001
        """Check compliance with sklearn estimators."""
        check(estimator)

    @pytest.mark.xfail(reason="invalid checks should fail")
    @pytest.mark.parametrize(
        "estimator, check, name",
        check_estimator(
            estimators=ESTIMATORS_TO_CHECK,
            valid=False,
            return_expected_failed_checks=expected_failed_checks,
        ),
    )
    def test_check_estimator_sklearn_invalid(estimator, check, name):  # noqa: ARG001
        """Check compliance with sklearn estimators."""
        check(estimator)

else:

    @parametrize_with_checks(
        estimators=ESTIMATORS_TO_CHECK,
        expected_failed_checks=expected_failed_checks,
    )
    def test_check_estimator_sklearn(estimator, check):
        check(estimator)


def test_no_implemented_methods():
    """test that the methods are not implemented in the base class"""
    X = np.random.randint(0, 2, size=(100, 2, 1))
    estimator = LinearRegression()
    estimator.fit(X[:, 0], X[:, 1])
    basic_class = BasePerturbation(estimator=estimator)
    with pytest.raises(NotImplementedError):
        basic_class._permutation(X, features_group_id=None)


def test_check_importance():
    """test that the methods are not implemented in the base class"""
    X = np.random.randint(0, 2, size=(100, 2, 1))
    estimator = LinearRegression()
    estimator.fit(X[:, 0], X[:, 1])
    basic_class = BasePerturbation(estimator=estimator)
    basic_class.importances_ = []
    with pytest.raises(
        ValueError, match="The importance method has not yet been called."
    ):
        basic_class.importance_selection()


def test_base_cv_errors():
    """Test the errors of the BasePerturbationCV class"""
    with pytest.raises(
        ValueError,
        match="If estimators is a list, its length must be equal to the number of folds",
    ):
        BasePerturbationCV(
            estimators=[LinearRegression(), LinearRegression()], cv=KFold(n_splits=4)
        )

    X = np.random.randn(100, 5)
    y = np.random.randn(100)
    with pytest.raises(NotImplementedError):
        vim = BasePerturbationCV(estimators=LinearRegression(), cv=KFold(n_splits=2))
        vim.fit(X, y)
