import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


class RandomForestClassifierModified(RandomForestClassifier):
    def fit(self, X, y):
        self.y_ = y
        super().fit(X, y)

    def predict(self, X):
        super().predict(X)

    def predict_proba(self, X):
        super().predict_proba(X)

    def sample_same_leaf(self, X, y=None):
        if not (y is None):
            self.y_ = y
        rng = np.random.RandomState(self.get_params()["random_state"])
        # The number of samples to sample from in the next step
        random_samples = 10

        # Get the leaf indices for each sample in the input data
        leaf_indices = self.apply(X)

        # Initialize an array to store the predictions for each sample
        predictions = []

        # Loop over each sample in the input data
        for i in range(X.shape[0]):
            # Removing the row of the corresponding input sample
            leaf_indices_minus_i = np.delete(leaf_indices, i, axis=0)
            y_minus_i = np.delete(self.y_, i, axis=0)

            # The list of indices sampled from the same leaf of the input sample
            leaf_samples = []
            # Loop over each tree in the forest
            for j in range(self.n_estimators):
                # Find the samples that fall into the same leaf node for this tree
                samples_in_leaf = np.where(
                    leaf_indices_minus_i[:, j] == leaf_indices[i, j]
                )[0]

                # Append the samples to the list
                leaf_samples.append(
                    y_minus_i[rng.choice(samples_in_leaf, size=random_samples)]
                )

            predictions.append(leaf_samples)

        # Combine the predictions from all trees to make the final prediction
        return np.array(predictions)


class RandomForestRegressorModified(RandomForestRegressor):
    def fit(self, X, y):
        self.y_ = y
        super().fit(X, y)

    def predict(self, X):
        super().predict(X)

    def sample_same_leaf(self, X):
        rng = np.random.RandomState(self.get_params()["random_state"])
        # The number of samples to sample from in the next step
        random_samples = 10

        # Get the leaf indices for each sample in the input data
        leaf_indices = self.apply(X)

        # Initialize an array to store the predictions for each sample
        predictions = []

        # Loop over each sample in the input data
        for i in range(X.shape[0]):
            # Removing the row of the corresponding input sample
            leaf_indices_minus_i = np.delete(leaf_indices, i, axis=0)
            y_minus_i = np.delete(self.y_, i, axis=0)

            # The list of indices sampled from the same leaf of the input sample
            leaf_samples = []
            # Loop over each tree in the forest
            for j in range(self.n_estimators):
                # Find the samples that fall into the same leaf node for this tree
                samples_in_leaf = np.where(
                    leaf_indices_minus_i[:, j] == leaf_indices[i, j]
                )[0]

                # Append the samples to the list
                leaf_samples.append(
                    y_minus_i[rng.choice(samples_in_leaf, size=random_samples)]
                )

            predictions.append(leaf_samples)

        # Combine the predictions from all trees to make the final prediction
        return np.array(predictions)
