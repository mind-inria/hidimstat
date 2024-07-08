import warnings
from collections import Counter
import itertools
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder

from .RandomForestModified import (
    RandomForestClassifierModified,
    RandomForestRegressorModified,
)
from .utils import convert_predict_proba, ordinal_encode, sample_predictions

warnings.filterwarnings("ignore")


def joblib_compute_conditional(
    p_col,
    n_sample,
    estimator,
    type_predictor,
    importance_estimator,
    X_test_list,
    y_test,
    prob_type,
    org_pred,
    seed=None,
    dict_cont={},
    dict_nom={},
    X_nominal=None,
    list_nominal={},
    encoder={},
    proc_col=None,
    index_i=None,
    sub_groups=None,
    group_stacking=False,
    list_seeds=None,
    Perm=False,
    output_dim=1,
    verbose=0,
):
    """This function applies the conditional approach for feature importance.
    Parameters
    ----------
    p_col: list
        The list of single variables/groups to compute the feature importance.
    n_sample: int
        The number of permutations/samples to loop
    estimator: scikit-learn compatible estimator, default=None
        The provided estimator for the prediction task (First block)
    type_predictor: string
        The provided predictor either the DNN learner or not.
        The DNN learner will use different inputs for the different
        sub-models especially while applying the stacking case.
    importance_estimator: scikit-learn compatible estimator, default=None
        The provided estimator for the importance task (Second block)
    X_test_list: list
        The list of inputs containing either one input or a number of inputs
        equal to the number of sub-models of the DNN learner.
    y_test: {array-like, sparse matrix}, shape (n_samples, n_output)
        The output test samples.
    prob_type: str, default='regression'
        A classification or a regression problem.
    org_pred: {array-like, sparse matrix}, shape (n_output, n_samples)
        The predictions using the original samples.
    seed: int, default=None
        Fixing the seeds of the random generator.
    dict_cont: dict, default={}
        The dictionary providing the indices of the continuous variables.
    dict_nom: dict, default={}
        The dictionary providing the indices of the categorical variables.
    X_nominal: {array-like}, default=None
        The dataframe of categorical variables without encoding.
    list_nominal: dict, default={}
        The dictionary of binary, nominal and ordinal variables.
    encoder: dict, default={}
        The dictionary of encoders for categorical variables.
    proc_col: int, default=None
        The processed column to print if verbose > 0.
    index_i: int, default=None
        The index of the current processed iteration.
    group_stacking: bool, default=False
        Apply the stacking-based method for the provided groups.
    list_seeds: list, default=None
        The list of seeds to fix the RandomState for reproducibility.
    Perm: bool, default=False
        The use of permutations or random sampling with CPI-DNN.
    output_dim:
    verbose: int, default=0
        If verbose > 0, the fitted iterations will be printed.
    """
    rng = np.random.RandomState(seed)

    res_ar = np.empty((n_sample, y_test.shape[0], output_dim))

    # A list of copied items to avoid any overlapping in the process
    current_X_test_list = [
        (lambda x: x.copy()[np.newaxis, ...] if len(x.shape) != 3 else x.copy())(
            X_test_el
        )
        for X_test_el in X_test_list
    ]

    el_ind = None
    # Conditioning on a subset of variables
    if sub_groups[1] is not None:
        if (proc_col + 1) in sub_groups[1].keys():
            # Get the indices of the provided variables/groups
            el = [i - 1 for i in sub_groups[1][proc_col + 1]]
            if not group_stacking:
                el = list(itertools.chain(*[sub_groups[0][i] for i in el]))
                el_ind = []
                for val in el:
                    if val in dict_nom.keys():
                        el_ind += dict_nom[val]
                    if val in dict_cont.keys():
                        el_ind += dict_cont[val]
                el_ind = list(set(el_ind))
            else:
                # Needs to be tested with multi-output neurones
                el_ind = el.copy()

    Res_col = [None] * len(current_X_test_list)
    X_col_pred = {
        "regression": [None] * len(current_X_test_list),
        "classification": None,
        "ordinal": None,
    }

    # Group partitioning
    grp_nom = [
        item
        for item in p_col
        if (item in dict_nom.keys())
        and (item in list_nominal["nominal"] + list_nominal["binary"])
    ]
    grp_ord = [
        item for item in p_col if (item in dict_nom.keys()) and (item not in grp_nom)
    ]
    grp_cont = [item for item in p_col if item not in (grp_nom + grp_ord)]

    # If modified Random Forest is applied
    pred_mod_RF = {
        "regression": [None] * len(current_X_test_list),
        "classification": [None],
        "ordinal": [None] * len(grp_ord),
    }

    p_col_n = {"regression": [], "classification": [], "ordinal": []}
    if not group_stacking:
        for val in p_col:
            if val in grp_nom:
                p_col_n["classification"] += dict_nom[val]
            if val in grp_cont:
                p_col_n["regression"] += dict_cont[val]
            if val in grp_ord:
                p_col_n["ordinal"] += dict_nom[val]
    else:
        p_col_n["regression"] = p_col

    # Dictionary of booleans checking for the encountered type of group
    var_type = {
        "regression": False,
        "classification": False,
        "ordinal": False,
    }

    X_col_new = {
        "regression": None,
        "classification": None,
        "ordinal": None,
    }
    output = {"regression": None, "classification": None, "ordinal": None}
    importance_models = {
        "regression": None,
        "classification": None,
        "ordinal": None,
    }

    if importance_estimator is None:
        importance_models["regression"] = RandomForestRegressor(
            random_state=seed, max_depth=5
        )
        importance_models["classification"] = RandomForestClassifier(
            random_state=seed, max_depth=5
        )
        importance_models["ordinal"] = RandomForestClassifier(
            random_state=seed, max_depth=5
        )
    elif importance_estimator == "Mod_RF":
        importance_models["regression"] = RandomForestRegressorModified(
            random_state=seed,
            min_samples_leaf=10,
        )
        importance_models["classification"] = RandomForestClassifierModified(
            random_state=seed, min_samples_leaf=10
        )
        importance_models["ordinal"] = RandomForestClassifierModified(
            random_state=seed, min_samples_leaf=10
        )
    else:
        importance_models = importance_estimator.copy()

    # Checking for pure vs hybrid groups
    if len(grp_cont) > 0:
        var_type["regression"] = True
    if len(grp_nom) > 0:
        var_type["classification"] = True
    if len(grp_ord) > 0:
        var_type["ordinal"] = True

    # All the variables of the same group will be removed simultaneously so it can be
    # predicted conditional on the remaining variables
    if importance_estimator != "Mod_RF":
        if var_type["regression"]:
            for counter_test, X_test_comp in enumerate(current_X_test_list):
                if el_ind is not None:
                    X_test_minus_idx = np.copy(X_test_comp)[..., el_ind]
                else:
                    X_test_minus_idx = np.delete(
                        np.copy(X_test_comp),
                        p_col_n["regression"]
                        + p_col_n["classification"]
                        + p_col_n["ordinal"],
                        -1,
                    )

                # Nb of y outputs x Nb of samples x Nb of regression outputs
                output["regression"] = X_test_comp[..., p_col_n["regression"]]
                X_col_pred["regression"][counter_test] = []
                for cur_output_ind in range(X_test_minus_idx.shape[0]):
                    importance_models["regression"] = hypertune_predictor(
                        importance_models["regression"],
                        X_test_minus_idx[cur_output_ind, ...],
                        output["regression"][cur_output_ind, ...],
                        param_grid={"max_depth": [2, 5, 10]},
                    )
                    importance_models["regression"].fit(
                        X_test_minus_idx[cur_output_ind, ...],
                        output["regression"][cur_output_ind, ...],
                    )
                    X_col_pred["regression"][counter_test].append(
                        importance_models["regression"].predict(
                            X_test_minus_idx[cur_output_ind, ...]
                        )
                    )

                X_col_pred["regression"][counter_test] = np.array(
                    X_col_pred["regression"][counter_test]
                )
                X_col_pred["regression"][counter_test] = X_col_pred["regression"][
                    counter_test
                ].reshape(output["regression"].shape)
                Res_col[counter_test] = (
                    output["regression"] - X_col_pred["regression"][counter_test]
                )

        # The loop doesn't include the classification part because the input across the different DNN sub-models
        # is the same without the stacking part (where extra sub-linear layers are used), therefore identical inputs
        # won't need looping classification process. This is not the case with the regression part.
        if var_type["classification"]:
            if el_ind is not None:
                X_test_minus_idx = np.copy(current_X_test_list[0])[..., el_ind]
            else:
                X_test_minus_idx = np.delete(
                    np.copy(current_X_test_list[0]),
                    p_col_n["regression"]
                    + p_col_n["classification"]
                    + p_col_n["ordinal"],
                    -1,
                )
            output["classification"] = np.array(X_nominal[grp_nom])
            X_col_pred["classification"] = []
            for cur_output_ind in range(X_test_minus_idx.shape[0]):
                importance_models["classification"] = hypertune_predictor(
                    importance_models["classification"],
                    X_test_minus_idx[cur_output_ind, ...],
                    output["classification"],
                    param_grid={"max_depth": [2, 5, 10]},
                )
                importance_models["classification"].fit(
                    X_test_minus_idx[cur_output_ind, ...],
                    output["classification"],
                )
                X_col_pred["classification"].append(
                    importance_models["classification"].predict_proba(
                        X_test_minus_idx[cur_output_ind, ...]
                    )
                )

                if not isinstance(X_col_pred["classification"][0], list):
                    X_col_pred["classification"] = [X_col_pred["classification"]]

        if var_type["ordinal"]:
            if el_ind is not None:
                X_test_minus_idx = np.copy(current_X_test_list[0])[..., el_ind]
            else:
                X_test_minus_idx = np.delete(
                    np.copy(current_X_test_list[0]),
                    p_col_n["regression"]
                    + p_col_n["classification"]
                    + p_col_n["ordinal"],
                    -1,
                )
            output["ordinal"] = ordinal_encode(np.array(X_nominal[grp_ord]))
            X_col_pred["ordinal"] = []
            for cur_output_ind in range(X_test_minus_idx.shape[0]):
                current_prediction = []
                for cur_ordinal in output["ordinal"]:
                    # To solve the problem of having multiple input matrices
                    # with multiple ordinal outcomes
                    importance_models["ordinal"] = hypertune_predictor(
                        importance_models["ordinal"],
                        X_test_minus_idx[cur_output_ind, ...],
                        cur_ordinal,
                        param_grid={"max_depth": [2, 5, 10]},
                    )
                    importance_models["ordinal"].fit(
                        X_test_minus_idx[cur_output_ind, ...], cur_ordinal
                    )
                    probs = importance_models["ordinal"].predict_proba(
                        X_test_minus_idx[cur_output_ind, ...]
                    )
                    current_prediction.append(convert_predict_proba(np.array(probs)))
                X_col_pred["ordinal"].append(current_prediction)
    else:
        for counter_test, X_test_comp in enumerate(current_X_test_list):
            if var_type["regression"]:
                if el_ind is not None:
                    X_test_minus_idx = np.copy(X_test_comp)[..., el_ind]
                else:
                    X_test_minus_idx = np.delete(
                        np.copy(X_test_comp),
                        p_col_n["regression"]
                        + p_col_n["classification"]
                        + p_col_n["ordinal"],
                        -1,
                    )
                output["regression"] = X_test_comp[..., p_col_n["regression"]]
                for cur_output_ind in range(X_test_minus_idx.shape[0]):
                    importance_models["regression"] = hypertune_predictor(
                        importance_models["regression"],
                        X_test_minus_idx[cur_output_ind, ...],
                        output["regression"][cur_output_ind, ...],
                        param_grid={"min_samples_leaf": [10, 15, 20]},
                    )

                    importance_models["regression"].fit(
                        X_test_minus_idx[cur_output_ind, ...],
                        output["regression"][cur_output_ind, ...],
                    )
                    pred_mod_RF["regression"][counter_test] = importance_models[
                        "regression"
                    ].sample_same_leaf(X_test_minus_idx[cur_output_ind, ...])

        if var_type["classification"]:
            if el_ind is not None:
                X_test_minus_idx = np.copy(current_X_test_list[0])[..., el_ind]
            else:
                X_test_minus_idx = np.delete(
                    np.copy(current_X_test_list[0]),
                    p_col_n["regression"]
                    + p_col_n["classification"]
                    + p_col_n["ordinal"],
                    -1,
                )
            output["classification"] = np.array(X_nominal[grp_nom])
            for cur_output_ind in range(X_test_minus_idx.shape[0]):
                importance_models["classification"] = hypertune_predictor(
                    importance_models["classification"],
                    X_test_minus_idx[cur_output_ind, ...],
                    output["classification"],
                    param_grid={"min_samples_leaf": [10, 15, 20]},
                )
                importance_models["classification"].fit(
                    X_test_minus_idx[cur_output_ind, ...],
                    output["classification"],
                )
                pred_mod_RF["classification"] = importance_models[
                    "classification"
                ].sample_same_leaf(X_test_minus_idx[cur_output_ind, ...])

        if var_type["ordinal"]:
            if el_ind is not None:
                X_test_minus_idx = np.copy(current_X_test_list[0])[..., el_ind]
            else:
                X_test_minus_idx = np.delete(
                    np.copy(current_X_test_list[0]),
                    p_col_n["regression"]
                    + p_col_n["classification"]
                    + p_col_n["ordinal"],
                    -1,
                )
            output["ordinal"] = ordinal_encode(np.array(X_nominal[grp_ord]))

            for cur_output_ind in range(X_test_minus_idx.shape[0]):
                for cur_ordinal_ind, cur_ordinal in enumerate(output["ordinal"]):
                    importance_models["ordinal"] = hypertune_predictor(
                        importance_models["ordinal"],
                        X_test_minus_idx[cur_output_ind, ...],
                        cur_ordinal,
                        param_grid={"min_samples_leaf": [10, 15, 20]},
                    )

                    importance_models["ordinal"].fit(
                        X_test_minus_idx[cur_output_ind, ...], cur_ordinal
                    )
                    pred_mod_RF["ordinal"][cur_ordinal_ind] = importance_models[
                        "ordinal"
                    ].sample_same_leaf(
                        X_test_minus_idx[cur_output_ind, ...],
                        np.array(X_nominal[grp_ord])[:, [cur_ordinal_ind]],
                    )

    if prob_type in ("classification", "binary"):
        nonzero_cols = np.where(y_test.any(axis=0))[0]
        score = roc_auc_score(y_test[:, nonzero_cols], org_pred[:, nonzero_cols])
    else:
        score = (
            mean_absolute_error(y_test, org_pred),
            r2_score(y_test, org_pred),
        )

    for sample in range(n_sample):
        if verbose > 0:
            if index_i is not None:
                print(
                    f"Iteration/Fold:{index_i}, Processing col:{proc_col+1}, Sample:{sample+1}"
                )
            else:
                print(f"Processing col:{proc_col+1}")
        # Same shuffled indices across the sub-models items
        indices = np.arange(current_X_test_list[0].shape[1])
        if importance_estimator != "Mod_RF":
            rng = np.random.RandomState(list_seeds[sample])
            if Perm:
                rng.shuffle(indices)
            else:
                indices = rng.choice(indices, size=len(indices))

            for counter_test, X_test_comp in enumerate(current_X_test_list):
                if var_type["regression"]:
                    X_test_comp[..., p_col_n["regression"]] = (
                        X_col_pred["regression"][counter_test]
                        + Res_col[counter_test][:, indices, :]
                    )

                if var_type["classification"]:
                    # X_col_pred['classification'] is a list of arrays representing the probability of getting
                    # each value at the  corresponding variable (resp. to each observation)
                    for cur_output in X_col_pred["classification"]:
                        list_seed_cat = rng.randint(1e5, size=len(cur_output))
                        for cat_prob_ind in range(len(cur_output)):
                            rng_cat = np.random.RandomState(list_seed_cat[cat_prob_ind])
                            current_X_col_new = np.array(
                                [
                                    [
                                        rng_cat.choice(
                                            np.unique(
                                                X_nominal[[grp_nom[cat_prob_ind]]]
                                            ),
                                            size=1,
                                            p=cur_output[cat_prob_ind][i],
                                        )[0]
                                        for i in range(
                                            cur_output[cat_prob_ind].shape[0]
                                        )
                                    ]
                                ]
                            ).T
                            if grp_nom[cat_prob_ind] in encoder:
                                current_X_col_new = (
                                    encoder[grp_nom[cat_prob_ind]]
                                    .transform(current_X_col_new)
                                    .toarray()
                                    .astype("int32")
                                )
                            if cat_prob_ind == 0:
                                X_col_new["classification"] = current_X_col_new
                            else:
                                X_col_new["classification"] = np.concatenate(
                                    (
                                        X_col_new["classification"],
                                        current_X_col_new,
                                    ),
                                    axis=1,
                                )
                    X_test_comp[..., p_col_n["classification"]] = X_col_new[
                        "classification"
                    ]

                if var_type["ordinal"]:
                    for cur_output in X_col_pred["ordinal"]:
                        list_seed_ord = rng.randint(1e5, size=len(cur_output))
                        for ord_prob_ind, ord_prob in enumerate(cur_output):
                            rng_ord = np.random.RandomState(list_seed_ord[ord_prob_ind])
                            current_X_col_new = [None] * ord_prob.shape[0]
                            for i in range(ord_prob.shape[0]):
                                current_sample = [0] * ord_prob.shape[1]
                                for ind_prob, prob in enumerate(ord_prob[i]):
                                    current_sample[ind_prob] = rng_ord.choice(
                                        [0, 1], p=[1 - prob, prob]
                                    )
                                    if current_sample[ind_prob] == 0:
                                        break
                                current_X_col_new[i] = np.array(current_sample)
                                # while True:
                                #     current_X_col_new[i] = rng_ord.binomial(1, ord_prob[i], ord_prob.shape[1])
                                #     print(current_X_col_new[i])
                                #     if np.any(np.all(output['ordinal'][ord_prob_ind] == current_X_col_new[i], axis=1)):
                                #         break
                                ind_cat = np.where(current_X_col_new[i] == 1)[0]
                                current_X_col_new[i] = [
                                    np.unique(X_nominal[grp_ord[ord_prob_ind]])[
                                        len(ind_cat)
                                    ]
                                ]

                            current_X_col_new = np.array(current_X_col_new)
                            if ord_prob_ind == 0:
                                X_col_new["ordinal"] = current_X_col_new
                            else:
                                X_col_new["ordinal"] = np.concatenate(
                                    (
                                        X_col_new["ordinal"],
                                        current_X_col_new,
                                    ),
                                    axis=1,
                                )
                    X_test_comp[..., p_col_n["ordinal"]] = X_col_new["ordinal"]
        else:
            for counter_test, X_test_comp in enumerate(current_X_test_list):
                if var_type["regression"]:
                    X_test_comp[..., p_col_n["regression"]] = np.mean(
                        sample_predictions(
                            pred_mod_RF["regression"][counter_test],
                            list_seeds[sample],
                        ),
                        axis=1,
                    )

                if var_type["classification"]:
                    predictions = sample_predictions(
                        pred_mod_RF["classification"], list_seeds[sample]
                    )
                    max_val = lambda row: list(Counter(row).keys())[0]
                    for cat_prob_ind in range(predictions.shape[-1]):
                        class_col = np.array(
                            [[max_val(row)] for row in predictions[..., cat_prob_ind]]
                        )
                        if grp_nom[cat_prob_ind] in encoder:
                            current_X_col_new = (
                                encoder[grp_nom[cat_prob_ind]]
                                .transform(class_col)
                                .toarray()
                                .astype("int32")
                            )
                        else:
                            current_X_col_new = class_col
                        if cat_prob_ind == 0:
                            X_col_new["classification"] = current_X_col_new
                        else:
                            X_col_new["classification"] = np.concatenate(
                                (
                                    X_col_new["classification"],
                                    current_X_col_new,
                                ),
                                axis=1,
                            )
                    X_test_comp[..., p_col_n["classification"]] = X_col_new[
                        "classification"
                    ]

                if var_type["ordinal"]:
                    max_val = lambda row: list(Counter(row).keys())[0]
                    for cur_output_ind in range(len(pred_mod_RF["ordinal"])):
                        predictions = sample_predictions(
                            pred_mod_RF["ordinal"][cur_output_ind],
                            list_seeds[sample],
                        )
                        class_ord = np.array(
                            [[max_val(row)] for row in predictions[..., 0]]
                        )
                        if cur_output_ind == 0:
                            X_col_new["ordinal"] = class_ord
                        else:
                            X_col_new["ordinal"] = np.concatenate(
                                (X_col_new["ordinal"], class_ord), axis=1
                            )
                    X_test_comp[..., p_col_n["ordinal"]] = X_col_new["ordinal"]

        if prob_type == "regression":
            if type_predictor == "DNN":
                pred_i = estimator.predict(current_X_test_list, scale=False)
            else:
                pred_i = estimator.predict(current_X_test_list[0].squeeze())

            # Convert to the (n_samples x n_outputs) format
            if len(pred_i.shape) != 2:
                pred_i = pred_i.reshape(-1, 1)

            res_ar[sample, ::] = (y_test - pred_i) ** 2 - (y_test - org_pred) ** 2
        else:
            if type_predictor == "DNN":
                pred_i = estimator.predict_proba(current_X_test_list, scale=False)
            else:
                pred_i = convert_predict_proba(
                    estimator.predict_proba(current_X_test_list[0].squeeze())
                )

            pred_i = np.clip(pred_i, 1e-10, 1 - 1e-10)
            org_pred = np.clip(org_pred, 1e-10, 1 - 1e-10)
            res_ar[sample, :, 0] = -np.sum(y_test * np.log(pred_i), axis=1) + np.sum(
                y_test * np.log(org_pred), axis=1
            )

    return res_ar, score


def joblib_compute_permutation(
    p_col,
    perm,
    estimator,
    type_predictor,
    X_test_list,
    y_test,
    prob_type,
    org_pred,
    dict_cont={},
    dict_nom={},
    proc_col=None,
    index_i=None,
    group_stacking=False,
    random_state=None,
    verbose=0,
):
    """This function applies the permutation feature importance (PFI).

    Parameters
    ----------
    p_col: list
        The list of single variables/groups to compute the feature importance.
    perm: int
        The current processed permutation (also to print if verbose > 0).
    estimator: scikit-learn compatible estimator, default=None
        The provided estimator for the prediction task (First block).
    type_predictor: string
        The provided predictor either the DNN learner or not.
        The DNN learner will use different inputs for the different
        sub-models especially while applying the stacking case.
    X_test_list: list
        The list of inputs containing either one input or a number of inputs
        equal to the number of sub-models of the DNN learner.
    y_test: {array-like, sparse matrix}, shape (n_samples, n_output)
        The output test samples.
    prob_type: str, default='regression'
        A classification or a regression problem.
    org_pred: {array-like, sparse matrix}, shape (n_output, n_samples)
        The predictions using the original samples.
    dict_cont: dict, default={}
        The dictionary providing the indices of the continuous variables.
    dict_nom: dict, default={}
        The dictionary providing the indices of the categorical variables.
    proc_col: int, default=None
        The processed column to print if verbose > 0.
    index_i: int, default=None
        The index of the current processed iteration.
    group_stacking: bool, default=False
        Apply the stacking-based method for the provided groups.
    random_state: int, default=None
        Fixing the seeds of the random generator.
    verbose: int, default=0
        If verbose > 0, the fitted iterations will be printed.
    """
    rng = np.random.RandomState(random_state)

    if verbose > 0:
        if index_i is not None:
            print(
                f"Iteration/Fold:{index_i}, Processing col:{proc_col+1}, Permutation:{perm+1}"
            )
        else:
            print(f"Processing col:{proc_col+1}, Permutation:{perm+1}")

    # A list of copied items to avoid any overlapping in the process
    current_X_test_list = [X_test_el.copy() for X_test_el in X_test_list]
    indices = np.arange(current_X_test_list[0].shape[-2])
    rng.shuffle(indices)

    if not group_stacking:
        p_col_new = []
        for val in p_col:
            if val in dict_nom.keys():
                p_col_new += dict_nom[val]
            else:
                p_col_new += dict_cont[val]
    else:
        p_col_new = p_col

    for X_test_comp in current_X_test_list:
        X_test_comp[..., p_col_new] = X_test_comp[..., p_col_new].take(indices, axis=-2)

    if prob_type == "regression":
        score = (
            mean_absolute_error(y_test, org_pred),
            r2_score(y_test, org_pred),
        )

        if type_predictor == "DNN":
            pred_i = estimator.predict(current_X_test_list, scale=False)
        else:
            pred_i = estimator.predict(current_X_test_list[0])

        # Convert to the (n_samples x n_outputs) format
        if len(pred_i.shape) != 2:
            pred_i = pred_i.reshape(-1, 1)

        res = (y_test - pred_i) ** 2 - (y_test - org_pred) ** 2
    else:
        nonzero_cols = np.where(y_test.any(axis=0))[0]
        score = roc_auc_score(y_test[:, nonzero_cols], org_pred[:, nonzero_cols])
        if type_predictor == "DNN":
            pred_i = estimator.predict_proba(current_X_test_list, scale=False)
        else:
            pred_i = convert_predict_proba(
                estimator.predict_proba(current_X_test_list[0])
            )

        pred_i = np.clip(pred_i, 1e-10, 1 - 1e-10)
        org_pred = np.clip(org_pred, 1e-10, 1 - 1e-10)
        res = -np.sum(y_test * np.log(pred_i), axis=1) + np.sum(
            y_test * np.log(org_pred), axis=1
        )
    return res, score


def hypertune_predictor(estimator, X, y, param_grid):
    grid_search = GridSearchCV(estimator, param_grid=param_grid, cv=2)
    grid_search.fit(X, y)
    return grid_search.best_estimator_
