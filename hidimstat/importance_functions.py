from Dnn_learner_single import DNN_learner_single
import numpy as np
from scipy.stats import ttest_1samp
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder


def compute_loco(X, y, ntree=100, seed=2021, prob_type="regression", dnn=True):
    """
    This function implements the Leave-One-Covariate-Out (LOCO) method for
    variable importance
    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        The training input samples.
    """
    y = np.array(y)
    dict_vals = {"val_imp": [], "p_value": []}
    if dnn:
        clf_rf = DNN_learner_single(
                prob_type=prob_type,
                do_hyper=True,
                random_state=2023,
                verbose=0,
            )
    else:
        if prob_type == "classification":
            clf_rf = RandomForestClassifier(n_estimators=ntree,
                                            random_state=seed)
        else: 
            clf_rf = RandomForestRegressor(n_estimators=ntree,
                                           random_state=seed)

    rng = np.random.RandomState(seed)
    train_ind = rng.choice(X.shape[0], int(X.shape[0] * 0.8), replace=False)
    test_ind = np.array([i for i in range(X.shape[0]) if i not in train_ind])

    # Full Model
    clf_rf.fit(X.iloc[train_ind, :], y[train_ind])

    if prob_type == "regression":
        loss = (y[test_ind] - np.ravel(clf_rf.predict(X.iloc[test_ind, :])))**2
    else:
        y_test = (
            OneHotEncoder(handle_unknown="ignore")
            .fit_transform(y[test_ind].reshape(-1, 1))
            .toarray()
        )

        loss = -np.sum(
            y_test * np.log(clf_rf.predict_proba(X.iloc[test_ind, :])), axis=1
        )

    # Retrain model

    for col in range(X.shape[1]):
        if dnn:
            clf_rf2 = DNN_learner_single(
                    prob_type=prob_type,
                    do_hyper=True,
                    random_state=2023,
                    verbose=0,
                )
        else:
            if prob_type == "classification":
                clf_rf2 = RandomForestClassifier(n_estimators=ntree,
                                                 random_state=seed)
            else: 
                clf_rf2 = RandomForestRegressor(n_estimators=ntree,
                                                random_state=seed)
            
        print(f"Processing col: {col+1}")
        X_minus_idx = np.delete(np.copy(X), col, -1)
        clf_rf2.fit(X_minus_idx[train_ind, :], y[train_ind])

        if prob_type == "regression":
            loss0 = (y[test_ind] - np.ravel(clf_rf2.predict(X_minus_idx[test_ind, :])))**2
        else:
            loss0 = np.sum(
                y_test * np.log(clf_rf2.predict_proba(X_minus_idx[test_ind, :])), axis=1
            )
        delta = loss0 - loss

        t_statistic, p_value = ttest_1samp(delta, 0, alternative="greater")
        if np.isnan(t_statistic):
            t_statistic = 0
        if np.isnan(p_value):
            p_value = 1
        dict_vals["val_imp"].append(np.mean(delta))
        dict_vals["p_value"].append(p_value)

    return dict_vals
