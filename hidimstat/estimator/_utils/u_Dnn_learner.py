
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.preprocessing import StandardScaler



def create_X_y(
    X,
    y,
    sampling_with_repetition=True,
    split_percentage=0.8,
    problem_type="regression",
    list_continuous=None,
    random_state=None,
):
    """
    Create train/valid split of input data X and target variable y

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The input samples before the splitting process.
    y : ndarray, shape (n_samples, )
        The output samples before the splitting process.
    sampling_with_repetition : bool, default=True
        Sampling with repetition the train part of the train/valid scheme under
        the training set. The number of training samples in train is equal to
        the number of instances in the training set.
    split_percentage : float, default=0.8
        The training/validation cut for the provided data.
    problem_type : str, default='regression'
        A classification or a regression problem.
    list_continuous : list, default=[]
        The list of continuous variables.
    random_state : int, default=2023
        Fixing the seeds of the random generator.

    Returns
    -------
    X_train_scaled : {array-like, sparse matrix} of shape (n_train_samples, n_features)
        The sampling_with_repetitionped training input samples with scaled continuous variables.
    y_train_scaled : {array-like} of shape (n_train_samples, )
        The sampling_with_repetitionped training output samples scaled if continous.
    X_validation_scaled : {array-like, sparse matrix} of shape (n_validation_samples, n_features)
        The validation input samples with scaled continuous variables.
    y_validation_scaled : {array-like} of shape (n_validation_samples, )
        The validation output samples scaled if continous.
    X_scaled : {array-like, sparse matrix} of shape (n_samples, n_features)
        The original input samples with scaled continuous variables.
    y_validation : {array-like} of shape (n_samples, )
        The original output samples with validation indices.
    scaler_x : Scikit-learn StandardScaler
        The standard scaler encoder for the continuous variables of the input.
    scaler_y : Scikit-learn StandardScaler
        The standard scaler encoder for the output if continuous.
    valid_ind : list
        The list of indices of the validation set.
    """
    rng = np.random.RandomState(random_state)
    scaler_x, scaler_y = StandardScaler(), StandardScaler()
    n = X.shape[0]

    if sampling_with_repetition:
        train_ind = rng.choice(n, n, replace=True)
    else:
        train_ind = rng.choice(
            n, size=int(np.floor(split_percentage * n)), replace=False
        )
    valid_ind = np.array([ind for ind in range(n) if ind not in train_ind])

    X_train, X_validation = X[train_ind], X[valid_ind]
    y_train, y_validation = y[train_ind], y[valid_ind]

    # Scaling X and y
    X_train_scaled = X_train.copy()
    X_validation_scaled = X_validation.copy()
    X_scaled = X.copy()

    if len(list_continuous) > 0:
        X_train_scaled[:, list_continuous] = scaler_x.fit_transform(
            X_train[:, list_continuous]
        )
        X_validation_scaled[:, list_continuous] = scaler_x.transform(
            X_validation[:, list_continuous]
        )
        X_scaled[:, list_continuous] = scaler_x.transform(X[:, list_continuous])
    if problem_type == "regression":
        y_train_scaled = scaler_y.fit_transform(y_train)
        y_validation_scaled = scaler_y.transform(y_validation)
    else:
        y_train_scaled = y_train.copy()
        y_validation_scaled = y_validation.copy()

    return (
        X_train_scaled,
        y_train_scaled,
        X_validation_scaled,
        y_validation_scaled,
        X_scaled,
        y_validation,
        scaler_x,
        scaler_y,
        valid_ind,
    )


def sigmoid(x):
    """
    This function applies the sigmoid function element-wise to the input array x
    """
    return 1 / (1 + np.exp(-x))


def softmax(x):
    """
    This function applies the softmax function element-wise to the input array x
    """
    # Ensure numerical stability by subtracting the maximum value of x from each element of x
    # This prevents overflow errors when exponentiating large numbers
    x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def relu(x):
    """
    This function applies the relu function element-wise to the input array x
    """
    return (abs(x) + x) / 2


def ordinal_encode(y):
    """
    This function encodes the ordinal variable with a special gradual encoding storing also
    the natural order information.
    """
    list_y = []
    for y_col in range(y.shape[-1]):
        # Retrieve the unique values
        unique_vals = np.unique(y[:, y_col])
        # Mapping each unique value to its corresponding index
        mapping_dict = {}
        for i, val in enumerate(unique_vals):
            mapping_dict[val] = i + 1
        # create a zero-filled array for the ordinal encoding
        y_ordinal = np.zeros((len(y[:, y_col]), len(set(y[:, y_col]))))
        # set the appropriate indices to 1 for each ordinal value and all lower ordinal values
        for ind_el, el in enumerate(y[:, y_col]):
            y_ordinal[ind_el, np.arange(mapping_dict[el])] = 1
        list_y.append(y_ordinal[:, 1:])

    return list_y


def joblib_ensemble_dnnet(
    X,
    y,
    problem_type="regression",
    activation_outcome=None,
    list_continuous=None,
    list_grps=None,
    sampling_with_repetition=False,
    split_percentage=0.8,
    group_stacking=False,
    input_dimensions=None,
    n_epoch=200,
    batch_size=32,
    beta1=0.9,
    beta2=0.999,
    lr=1e-3,
    l1_weight=1e-2,
    l2_weight=1e-2,
    epsilon=1e-8,
    random_state=None,
):
    """
    This function implements the ensemble learning of the sub-DNN models

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_train_samples, n_features)
        The input samples.
    y : array-like of shape (n_train_samples,) or (n_train_samples, n_outputs)
        The target values (class labels in classification, real numbers in
        regression).
    problem_type : str, default='regression'
        A classification or a regression problem.
    activation_outcome : str, default=None
        The activation function to apply in the outcome layer, "softmax" for
        classification and "sigmoid" for both ordinal and binary cases.
    list_continuous : list, default=None
        The list of continuous variables.
    list_grps : list of lists, default=None
        A list collecting the indices of the groups' variables
        while applying the stacking method.
    sampling_with_repetition : bool, default=True
        Application of sampling_with_repetition sampling for the training set.
    split_percentage : float, default=0.8
        The training/validation cut for the provided data.
    group_stacking : bool, default=False
        Apply the stacking-based method for the provided groups.
    input_dimensions : list, default=None
        The cumsum of inputs after the linear sub-layers.
    n_epoch : int, default=200
        The number of epochs for the DNN learner(s).
    batch_size : int, default=32
        The number of samples per batch for training.
    beta1 : float, default=0.9
        The exponential decay rate for the first moment estimates.
    beta2 : float, default=0.999
        The exponential decay rate for the second moment estimates.
    lr : float, default=1e-3
        The learning rate.
    l1_weight : float, default=1e-2
        The L1-regularization paramter for weight decay.
    l2_weight : float, default=0
        The L2-regularization paramter for weight decay.
    epsilon : float, default=1e-8
        A small constant added to the denominator to prevent division by zero.
    random_state : int, default=2023
        Fixing the seeds of the random generator.

    Returns
    -------
    current_model : list
        The parameters of the sub-DNN model
    scaler_x : list of Scikit-learn StandardScalers
        The scalers for the continuous input variables.
    scaler_y : Scikit-learn StandardScaler
        The scaler for the continuous output variable.
    pred_v : ndarray
        The predictions of the sub-DNN model.
    loss : float
        The loss score of the sub-DNN model.
    """

    pred_v = np.empty(X.shape[0])
    # Sampling and Train/Validate splitting
    (
        X_train_scaled,
        y_train_scaled,
        X_validation_scaled,
        y_validation_scaled,
        X_scaled,
        y_validation,
        scaler_x,
        scaler_y,
        valid_ind,
    ) = create_X_y(
        X,
        y,
        sampling_with_repetition=sampling_with_repetition,
        split_percentage=split_percentage,
        problem_type=problem_type,
        list_continuous=list_continuous,
        random_state=random_state,
    )

    current_model = dnn_net(
        X_train_scaled,
        y_train_scaled,
        X_validation_scaled,
        y_validation_scaled,
        problem_type=problem_type,
        n_epoch=n_epoch,
        batch_size=batch_size,
        beta1=beta1,
        beta2=beta2,
        lr=lr,
        l1_weight=l1_weight,
        l2_weight=l2_weight,
        epsilon=epsilon,
        list_grps=list_grps,
        group_stacking=group_stacking,
        input_dimensions=input_dimensions,
        random_state=random_state,
    )

    if not group_stacking:
        X_scaled_n = X_scaled.copy()
    else:
        X_scaled_n = np.zeros((X_scaled.shape[0], input_dimensions[-1]))
        for grp_ind in range(len(list_grps)):
            n_layer_stacking = len(current_model[3][grp_ind]) - 1
            curr_pred = X_scaled[:, list_grps[grp_ind]].copy()
            for ind_w_b in range(n_layer_stacking):
                if ind_w_b == 0:
                    curr_pred = relu(
                        X_scaled[:, list_grps[grp_ind]].dot(
                            current_model[3][grp_ind][ind_w_b]
                        )
                        + current_model[4][grp_ind][ind_w_b]
                    )
                else:
                    curr_pred = relu(
                        curr_pred.dot(current_model[3][grp_ind][ind_w_b])
                        + current_model[4][grp_ind][ind_w_b]
                    )
            X_scaled_n[
                :,
                list(
                    np.arange(input_dimensions[grp_ind], input_dimensions[grp_ind + 1])
                ),
            ] = (
                curr_pred.dot(current_model[3][grp_ind][n_layer_stacking])
                + current_model[4][grp_ind][n_layer_stacking]
            )

    n_layer = len(current_model[0]) - 1
    for j in range(n_layer):
        if j == 0:
            pred = relu(X_scaled_n.dot(current_model[0][j]) + current_model[1][j])
        else:
            pred = relu(pred.dot(current_model[0][j]) + current_model[1][j])

    pred = pred.dot(current_model[0][n_layer]) + current_model[1][n_layer]

    if problem_type not in ("classification", "binary"):
        if problem_type != "ordinal":
            pred_v = pred * scaler_y.scale_ + scaler_y.mean_
        else:
            pred_v = activation_outcome[problem_type](pred)
        loss = np.std(y_validation) ** 2 - mean_squared_error(
            y_validation, pred_v[valid_ind]
        )
    else:
        pred_v = activation_outcome[problem_type](pred)
        loss = log_loss(
            y_validation, np.ones(y_validation.shape) * np.mean(y_validation, axis=0)
        ) - log_loss(y_validation, pred_v[valid_ind])

    return (current_model, scaler_x, scaler_y, pred_v, loss)


def _initialize_weights(layer):
    if isinstance(layer, nn.Linear):
        layer.weight.data = (layer.weight.data.uniform_() - 0.5) * 0.2
        layer.bias.data = (layer.bias.data.uniform_() - 0.5) * 0.1


def _dataset_Loader(X, y, shuffle=False, batch_size=50):
    if y.shape[-1] == 2:
        y = y[:, [1]]
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(X).float(), torch.from_numpy(y).float()
    )

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle
    )
    return loader


class DNN(nn.Module):
    """
    Feedfoward Neural Network with 4 hidden layers
    """

    def __init__(
        self, input_dim, group_stacking, list_grps, output_dimension, problem_type
    ):
        super().__init__()
        if problem_type == "classification":
            self.accuracy = Accuracy(task="multiclass", num_classes=output_dimension)
        else:
            self.accuracy = Accuracy(task="binary")
        self.list_grps = list_grps
        self.group_stacking = group_stacking
        if group_stacking:
            self.layers_stacking = nn.ModuleList(
                [
                    nn.Linear(
                        in_features=len(grp),
                        out_features=input_dim[grp_ind + 1] - input_dim[grp_ind],
                    )
                    # nn.Sequential(
                    #     nn.Linear(
                    #         in_features=len(grp),
                    #         # out_features=max(1, int(0.1 * len(grp))),
                    #         out_features=input_dim[grp_ind + 1]
                    #         - input_dim[grp_ind],
                    #     ),
                    #     nn.ReLU(),
                    # nn.Linear(
                    #     in_features=max(1, int(0.1 * len(grp))),
                    #     out_features=input_dim[grp_ind + 1]
                    #     - input_dim[grp_ind],
                    # ),
                    # nn.ReLU(),
                    # nn.Linear(
                    #     in_features=max(1, int(0.1 * len(grp))),
                    #     out_features=input_dim[grp_ind + 1]
                    #     - input_dim[grp_ind],
                    # ),
                    # )
                    for grp_ind, grp in enumerate(list_grps)
                ]
            )
            input_dim = input_dim[-1]
        self.layers = nn.Sequential(
            # hidden layers
            nn.Linear(input_dim, 50),
            nn.ReLU(),
            nn.Linear(50, 40),
            nn.ReLU(),
            nn.Linear(40, 30),
            nn.ReLU(),
            nn.Linear(30, 20),
            nn.ReLU(),
            # output layer
            nn.Linear(20, output_dimension),
        )
        self.loss = 0

    def forward(self, x):
        if self.group_stacking:
            list_stacking = [None] * len(self.layers_stacking)
            for ind_layer, layer in enumerate(self.layers_stacking):
                list_stacking[ind_layer] = layer(x[:, self.list_grps[ind_layer]])
            x = torch.cat(list_stacking, dim=1)
        return self.layers(x)

    def training_step(self, batch, device, problem_type):
        X, y = batch[0].to(device), batch[1].to(device)
        y_pred = self(X)  # Generate predictions
        if problem_type == "regression":
            loss = F.mse_loss(y_pred, y)
        elif problem_type == "classification":
            loss = F.cross_entropy(y_pred, y)  # Calculate loss
        else:
            loss = F.binary_cross_entropy_with_logits(y_pred, y)
        return loss

    def validation_step(self, batch, device, problem_type):
        X, y = batch[0].to(device), batch[1].to(device)
        y_pred = self(X)  # Generate predictions
        if problem_type == "regression":
            loss = F.mse_loss(y_pred, y)
            return {
                "val_mse": loss,
                "batch_size": len(X),
            }
        else:
            if problem_type == "classification":
                loss = F.cross_entropy(y_pred, y)  # Calculate loss
            else:
                loss = F.binary_cross_entropy_with_logits(y_pred, y)
            acc = self.accuracy(y_pred, y.int())
            return {
                "val_loss": loss,
                "val_acc": acc,
                "batch_size": len(X),
            }

    def validation_epoch_end(self, outputs, problem_type):
        if problem_type in ("classification", "binary", "ordinal"):
            batch_losses = []
            batch_accs = []
            batch_sizes = []
            for x in outputs:
                batch_losses.append(x["val_loss"] * x["batch_size"])
                batch_accs.append(x["val_acc"] * x["batch_size"])
                batch_sizes.append(x["batch_size"])
            self.loss = torch.stack(batch_losses).sum().item() / np.sum(
                batch_sizes
            )  # Combine losses
            epoch_acc = torch.stack(batch_accs).sum().item() / np.sum(
                batch_sizes
            )  # Combine accuracies
            return {"val_loss": self.loss, "val_acc": epoch_acc}
        else:
            batch_losses = [x["val_mse"] * x["batch_size"] for x in outputs]
            batch_sizes = [x["batch_size"] for x in outputs]
            self.loss = torch.stack(batch_losses).sum().item() / np.sum(
                batch_sizes
            )  # Combine losses
            return {"val_mse": self.loss}

    def epoch_end(self, epoch, result):
        if len(result) == 2:
            print(
                "Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(
                    epoch + 1, result["val_loss"], result["val_acc"]
                )
            )
        else:
            print("Epoch [{}], val_mse: {:.4f}".format(epoch + 1, result["val_mse"]))


def _evaluate(model, loader, device, problem_type):
    outputs = [model.validation_step(batch, device, problem_type) for batch in loader]
    return model.validation_epoch_end(outputs, problem_type)


def dnn_net(
    X_train,
    y_train,
    X_validation,
    y_validation,
    problem_type="regression",
    n_epoch=200,
    batch_size=32,
    batch_size_validation=128,
    beta1=0.9,
    beta2=0.999,
    lr=1e-3,
    l1_weight=1e-2,
    l2_weight=1e-2,
    epsilon=1e-8,
    list_grps=None,
    group_stacking=False,
    input_dimensions=None,
    random_state=2023,
    verbose=0,
):
    """
    This function implements the training/validation process of the sub-DNN
    models

    Parameters
    ----------
    X_train : {array-like, sparse matrix} of shape (n_train_samples, n_features)
        The training input samples.
    y_train : {array-like} of shape (n_train_samples, )
        The training output samples.
    X_validation : {array-like, sparse matrix} of shape (n_validation_samples, n_features)
        The validation input samples.
    y_validation : {array-like} of shape (n_validation_samples, )
        The validation output samples.
    problem_type : str, default='regression'
        A classification or a regression problem.
    n_epoch : int, default=200
        The number of epochs for the DNN learner(s).
    batch_size : int, default=32
        The number of samples per batch for training.
    batch_size_validation : int, default=128
        The number of samples per batch for validation.
    beta1 : float, default=0.9
        The exponential decay rate for the first moment estimates.
    beta2 : float, default=0.999
        The exponential decay rate for the second moment estimates.
    lr : float, default=1e-3
        The learning rate.
    l1_weight : float, default=1e-2
        The L1-regularization paramter for weight decay.
    l2_weight : float, default=0
        The L2-regularization paramter for weight decay.
    epsilon : float, default=1e-8
        A small constant added to the denominator to prevent division by zero.
    list_grps : list of lists, default=None
        A list collecting the indices of the groups' variables
        while applying the stacking method.
    group_stacking : bool, default=False
        Apply the stacking-based method for the provided groups.
    input_dimensions : list, default=None
        The cumsum of inputs after the linear sub-layers.
    random_state : int, default=2023
        Fixing the seeds of the random generator.
    verbose : int, default=0
        If verbose > 0, the fitted iterations will be printed.
    """
    # Creating DataLoaders
    train_loader = _dataset_Loader(
        X_train,
        y_train,
        shuffle=True,
        batch_size=batch_size,
    )
    validate_loader = _dataset_Loader(
        X_validation, y_validation, batch_size=batch_size_validation
    )
    # Set the seed for PyTorch's random number generator
    torch.manual_seed(random_state)

    # Set the seed for PyTorch's CUDA random number generator(s), if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_state)
        torch.cuda.manual_seed_all(random_state)

    # Specify whether to use GPU or CPU
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    if problem_type in ("regression", "binary"):
        output_dimension = 1
    else:
        output_dimension = y_train.shape[-1]

    # DNN model
    input_dim = input_dimensions.copy() if group_stacking else X_train.shape[1]
    model = DNN(input_dim, group_stacking, list_grps, output_dimension, problem_type)
    model.to(device)
    # Initializing weights/bias
    model.apply(_initialize_weights)
    # Adam Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, betas=(beta1, beta2), eps=epsilon
    )

    best_loss = 1e100
    for epoch in range(n_epoch):
        # Training Phase
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            loss = model.training_step(batch, device, problem_type)

            loss.backward()
            optimizer.step()
            for name, param in model.named_parameters():
                if "bias" not in name:
                    # if name.split(".")[0] == "layers_stacking":
                    #     param.data -= l2_weight * param.data
                    # else:
                    param.data -= (
                        l1_weight * torch.sign(param.data) + l2_weight * param.data
                    )
        # Validation Phase
        model.eval()
        result = _evaluate(model, validate_loader, device, problem_type)
        if model.loss < best_loss:
            best_loss = model.loss
            dict_params = copy.deepcopy(model.state_dict())
        if verbose >= 2:
            model.epoch_end(epoch, result)

    best_weight = []
    best_bias = []
    best_weight_stack = [[].copy() for _ in range(len(list_grps))]
    best_bias_stack = [[].copy() for _ in range(len(list_grps))]

    for name, param in dict_params.items():
        if name.split(".")[0] == "layers":
            if name.split(".")[-1] == "weight":
                best_weight.append(param.numpy().T)
            if name.split(".")[-1] == "bias":
                best_bias.append(param.numpy()[np.newaxis, :])
        if name.split(".")[0] == "layers_stacking":
            curr_ind = int(name.split(".")[1])
            if name.split(".")[-1] == "weight":
                best_weight_stack[curr_ind].append(param.numpy().T)
            if name.split(".")[-1] == "bias":
                best_bias_stack[curr_ind].append(param.numpy()[np.newaxis, :])

    return [
        best_weight,
        best_bias,
        best_loss,
        best_weight_stack,
        best_bias_stack,
    ]