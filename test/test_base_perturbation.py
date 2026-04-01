import pytest
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.utils.estimator_checks import parametrize_with_checks

from hidimstat.base_perturbation import BasePerturbation, BasePerturbationCV

from .conftest import SKLEARN_LT_1_6, check_estimator, fitted_linear_regression

ESTIMATORS_TO_CHECK = [
    BasePerturbation(estimator=fitted_linear_regression()),
    BasePerturbation(estimator=LinearRegression()),
    BasePerturbationCV(estimators=LinearRegression(), cv=KFold(n_splits=2)),
    BasePerturbationCV(
        estimators=[fitted_linear_regression(), fitted_linear_regression()],
        cv=KFold(n_splits=2),
    ),
]


def expected_failed_checks(estimator):
    if isinstance(estimator, BasePerturbation):
        return {
            "check_estimator_sparse_tag": "TODO",
        }
    elif isinstance(estimator, BasePerturbationCV):
        failed_checks = {
            "check_dict_unchanged": "'fit' requires '_fit_single_split' to be implemented.",
            "check_dont_overwrite_parameters": "'fit' requires '_fit_single_split' to be implemented.",
            "check_dtype_object": "'fit' requires '_fit_single_split' to be implemented.",
            "check_estimator_sparse_array": "'fit' requires '_fit_single_split' to be implemented.",
            "check_estimator_sparse_matrix": "'fit' requires '_fit_single_split' to be implemented.",
            "check_estimator_sparse_tag": "'fit' requires '_fit_single_split' to be implemented.",
            "check_estimators_dtypes": "'fit' requires '_fit_single_split' to be implemented.",
            "check_estimators_fit_returns_self": "'fit' requires '_fit_single_split' to be implemented.",
            "check_estimators_pickle": "'fit' requires '_fit_single_split' to be implemented.",
            "check_estimators_nan_inf": "'fit' requires '_fit_single_split' to be implemented.",
            "check_estimators_overwrite_params": "'fit' requires '_fit_single_split' to be implemented.",
            "check_f_contiguous_array_estimator": "'fit' requires '_fit_single_split' to be implemented.",
            "check_fit_check_is_fitted": "'fit' requires '_fit_single_split' to be implemented.",
            "check_fit_idempotent": "'fit' requires '_fit_single_split' to be implemented.",
            "check_fit_score_takes_y": "'fit' requires '_fit_single_split' to be implemented.",
            "check_fit2d_1feature": "'fit' requires '_fit_single_split' to be implemented.",
            "check_fit2d_predict1d": "'fit' requires '_fit_single_split' to be implemented.",
            "check_n_features_in": "'fit' requires '_fit_single_split' to be implemented.",
            "check_n_features_in_after_fitting": "'fit' requires '_fit_single_split' to be implemented.",
            "check_methods_sample_order_invariance": "'fit' requires '_fit_single_split' to be implemented.",
            "check_methods_subset_invariance": "'fit' requires '_fit_single_split' to be implemented.",
            "check_pipeline_consistency": "'fit' requires '_fit_single_split' to be implemented.",
            "check_positive_only_tag_during_fit": "'fit' requires '_fit_single_split' to be implemented.",
            "check_readonly_memmap_input": "'fit' requires '_fit_single_split' to be implemented.",
        }
        if isinstance(estimator.estimators, list):
            # there are some extra failures when BasePerturbationCV has a list of estimators
            failed_checks |= {
                "check_fit2d_1sample": "'fit' requires '_fit_single_split' to be implemented.",
                "check_complex_data": "'fit' requires '_fit_single_split' to be implemented.",
                "check_estimators_empty_data_messages": "'fit' requires '_fit_single_split' to be implemented.",
            }
        return failed_checks


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


def test_no_implemented_methods(rng):
    """Test that the methods are not implemented in the base class"""
    X = rng.integers(0, 2, size=(100, 2, 1))
    estimator = LinearRegression()
    estimator.fit(X[:, 0], X[:, 1])
    basic_class = BasePerturbation(estimator=estimator)
    with pytest.raises(NotImplementedError):
        basic_class._permutation(X, features_group_id=None)


def test_check_importance(rng):
    """Test that the methods are not implemented in the base class"""
    X = rng.integers(0, 2, size=(100, 2, 1))
    estimator = LinearRegression()
    basic_class = BasePerturbation(estimator=estimator).fit(X[:, 0], X[:, 1])
    with pytest.raises(
        ValueError,
        match="The importance method need to be called before calling this method",
    ):
        basic_class.importance_selection()


def test_base_cv_errors(rng):
    """Test the errors of the BasePerturbationCV class"""
    basic_class = BasePerturbationCV(
        estimators=[LinearRegression(), LinearRegression()],
        cv=KFold(n_splits=4),
    )

    X = rng.random((100, 5))
    y = rng.random(100)

    with pytest.raises(
        ValueError,
        match="If estimators is a list, its length must be equal to the number of folds",
    ):
        basic_class.fit(X, y)

    vim = BasePerturbationCV(
        estimators=LinearRegression(), cv=KFold(n_splits=2)
    )
    with pytest.raises(NotImplementedError):
        vim.fit(X, y)
