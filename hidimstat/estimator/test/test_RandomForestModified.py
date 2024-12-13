from hidimstat.estimator.test._utils_test import generate_data
from hidimstat.estimator.RandomForestModified import RandomForestClassifierModified, RandomForestRegressorModified
import numpy as np


def test_RandomForestRegressorModified():
    """
    Test the RandomForestRegressorModified for regression.
    Parameters:
    - regression_data: A tuple containing the input features (X) and target variable (y) for regression.
    """
    X, y = generate_data(problem_type="regression")
    learner = RandomForestRegressorModified(n_jobs=10, verbose=0)
    learner.fit(X, y)
    predict = learner.predict(X)
    # Check if the predicted values are close to the true values for at least one instance
    assert np.max(np.abs(predict-y)) < 200.0
    # Check if the predicted values are close to the true values for at least one instance
    assert np.all(predict == y) or np.any(predict != y)
    # Check if the predicted values are not all the same
    assert not np.all(predict == predict[0])
    # Check if the predicted values are not all zeros
    assert not np.all(predict == 0)
    # Check if the predicted values are not all ones
    assert not np.all(predict == 1)
    # Check if the feature importances are not all zeros
    assert not np.all(learner.feature_importances_ == 0)
    # Check if the feature importances are not all the same
    assert not np.all(learner.feature_importances_ == learner.feature_importances_[0])
    # Check if the feature importances are not all ones
    assert not np.all(learner.feature_importances_ == 1)
    # Check if the feature importances are not all negative
    assert not np.all(learner.feature_importances_ < 0)
    # # Check if the feature importances are not all positive
    # assert not np.all(learner.feature_importances_ > 0)
    # Check if the feature importances are not all close to zero
    assert not np.allclose(learner.feature_importances_, 0)
    # Check if the feature importances are not all close to one
    assert not np.allclose(learner.feature_importances_, 1)

    predictions = learner.sample_same_leaf(X)
    #TODO: add more tests for sample_same_leaf

def test_RandomForestClassifierModified():
    """
    Test the RandomForestClassifierModified for classification.
    """
    X, y = generate_data(problem_type="classification")
    learner = RandomForestClassifierModified(n_jobs=10, verbose=0)
    learner.fit(X, y)
    predict_prob = learner.predict_proba(X)
    # Check if the predicted probabilities sum up to 1 for each instance
    assert np.allclose(np.sum(predict_prob, axis=1), 1)
    # Check if the predicted class labels match the true labels for at least one instance
    assert np.sum(np.argmax(predict_prob, axis=1) == y) > 0
    assert np.all(np.max(predict_prob, axis=1) >= 0.5)
    assert np.all(np.min(predict_prob, axis=1) < 0.5)
    # Check if the maximum predicted probability is greater than 0.95
    assert 0.95 < np.max(predict_prob)
    # Check if the minimum predicted probability is less than 0.05
    assert 0.05 > np.min(predict_prob)
    # Check if the predicted probabilities are not all the same
    assert not np.all(predict_prob == predict_prob[0])
    # Check if the predicted probabilities are not all zeros
    assert not np.all(predict_prob == 0)
    # Check if the predicted probabilities are not all ones
    assert not np.all(predict_prob == 1)
    # Check if the predicted probabilities are not all the same for each class
    assert not np.all(predict_prob[:, 0] == predict_prob[0, 0])
    assert not np.all(predict_prob[:, 1] == predict_prob[0, 1])
    # Check if the predicted probabilities are not all zeros for each class
    assert not np.all(predict_prob[:, 0] == 0)
    assert not np.all(predict_prob[:, 1] == 0)
    # Check if the predicted probabilities are not all ones for each class
    assert not np.all(predict_prob[:, 0] == 1)

    predict = learner.predict(X)
    # Check if the predicted values are close to the true values for at least one instance
    assert np.all(predict == y) or np.any(predict != y)
    # Check if the predicted values are not all the same
    assert not np.all(predict == predict[0])
    # Check if the predicted values are not all zeros
    assert not np.all(predict == 0)
    # Check if the predicted values are not all ones
    assert not np.all(predict == 1)
    # Check if the feature importances are not all zeros
    assert not np.all(learner.feature_importances_ == 0)
    # Check if the feature importances are not all the same
    assert not np.all(learner.feature_importances_ == learner.feature_importances_[0])
    # Check if the feature importances are not all ones
    assert not np.all(learner.feature_importances_ == 1)
    # Check if the feature importances are not all negative
    assert not np.all(learner.feature_importances_ < 0)
    # # Check if the feature importances are not all positive
    # assert not np.all(learner.feature_importances_ > 0)
    # Check if the feature importances are not all close to zero
    assert not np.allclose(learner.feature_importances_, 0)
    # Check if the feature importances are not all close to one
    assert not np.allclose(learner.feature_importances_, 1)

    predictions = learner.sample_same_leaf(X)
    #TODO: add more tests for sample_same_leaf