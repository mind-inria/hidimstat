from hidimstat.Dnn_learner_single import DNN_learner_single
import numpy as np
from scipy.stats import ttest_1samp
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder


def compute_loco(X, y, ntree=100, problem_type="regression", use_dnn=True, seed=2024):
    """
    This function implements the Leave-One-Covariate-Out (LOCO) method for
    variable importance

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        The training input samples.
    y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        The target values (class labels in classification, real numbers in
        regression).
    ntree : int, default=100
        The number of trees when using the Random Forest estimator.
    problem_type : str, default='regression'
        A classification or a regression problem.
    use_dnn : bool, default=True
        The Deep Neural Network or Random Forest estimator.
    seed : int, default=2024
        Fixing the seeds of the random generator.

    Returns
    -------
    dict_vals : dict
        A dictionary containing the importance scores (importance_score) and
        p-values (p_value).
    """
    y = np.array(y)
    dict_vals = {"importance_score": [], "p_value": []}
    dict_encode_outcome = {"regression": False, "classification": True}

    if use_dnn:
        clf_rf_full = DNN_learner_single(
            encode=dict_encode_outcome[problem_type],
            problem_type=problem_type,
            do_hypertuning=True,
            random_state=seed,
            verbose=0,
        )
    else:
        if problem_type == "classification":
            clf_rf_full = RandomForestClassifier(n_estimators=ntree, random_state=seed)
        else:
            clf_rf_full = GridSearchCV(
                RandomForestRegressor(n_estimators=ntree, random_state=seed),
                param_grid=[{"max_depth": [2, 5, 10]}],
                cv=5,
            )

    rng = np.random.RandomState(seed)
    train_ind = rng.choice(X.shape[0], int(X.shape[0] * 0.8), replace=False)
    test_ind = np.array([i for i in range(X.shape[0]) if i not in train_ind])

    # Full Model
    clf_rf_full.fit(X.iloc[train_ind, :], y[train_ind])

    if problem_type == "regression":
        loss_full = (
            y[test_ind] - np.ravel(clf_rf_full.predict(X.iloc[test_ind, :]))
        ) ** 2
    else:
        y_test = (
            OneHotEncoder(handle_unknown="ignore")
            .fit_transform(y[test_ind].reshape(-1, 1))
            .toarray()
        )

        loss_full = -np.sum(
            y_test * np.log(clf_rf_full.predict_proba(X.iloc[test_ind, :]) + 1e-100),
            axis=1,
        )

    # Retrain model
    for col in range(X.shape[1]):
        if use_dnn:
            clf_rf_retrain = DNN_learner_single(
                encode=dict_encode_outcome[problem_type],
                problem_type=problem_type,
                do_hypertuning=True,
                random_state=seed,
                verbose=0,
            )
        else:
            if problem_type == "classification":
                clf_rf_retrain = RandomForestClassifier(
                    n_estimators=ntree, random_state=seed
                )
            else:
                clf_rf_retrain = RandomForestRegressor(
                    n_estimators=ntree, random_state=seed
                )

        print(f"Processing col: {col+1}")
        X_minus_idx = np.delete(np.copy(X), col, -1)
        clf_rf_retrain.fit(X_minus_idx[train_ind, :], y[train_ind])

        if problem_type == "regression":
            loss_retrain = (
                y[test_ind] - np.ravel(clf_rf_retrain.predict(X_minus_idx[test_ind, :]))
            ) ** 2
        else:
            loss_retrain = np.sum(
                y_test
                * np.log(
                    clf_rf_retrain.predict_proba(X_minus_idx[test_ind, :]) + 1e-100
                ),
                axis=1,
            )
        delta = loss_retrain - loss_full

        t_statistic, p_value = ttest_1samp(delta, 0, alternative="greater")
        if np.isnan(t_statistic):
            t_statistic = 0
        if np.isnan(p_value):
            p_value = 1
        dict_vals["importance_score"].append(np.mean(delta))
        dict_vals["p_value"].append(p_value)

    return dict_vals
