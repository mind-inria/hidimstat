"""
Controlled variable selection: why and how?
=======================================
"""

# %% [markdown]
#
# In this example, we explore the basics of variable selection and illustrate the need to statistically control the amount of faslely selected variables. We start off by loading the Wisconsin breast cancer dataset:

# %%
import numpy as np

# Set random seed for reproducibility
seed = 99
np.random.seed(seed)

# Load and standardize the breast cancer dataset
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

data = load_breast_cancer()
X = data.data
y = data.target

X = StandardScaler().fit_transform(X)

print(X.shape)
n, p = X.shape

# %% [markdown]
# There are 569 samples and 30 features that correspond to tumor attributes. The downstream task is to classify tumors as benign or malignant.

# %%
feature_names = [str(name) for name in data.feature_names]
feature_names[:10]

# %%
data.target_names

#############################################################################
# Step 1: Selecting variables with the Lasso
# ------------------------------------------
#
# We want to select variables that are relevant to the outcome, i.e. tumor charateristics that are associated with tumor malignance. We start off by applying a classical method using Lasso logistic regression and retaining variables with non-null coefficients:

# %%
from sklearn.linear_model import LogisticRegressionCV

clf = LogisticRegressionCV(penalty="l1", solver="liblinear", max_iter=int(1e4))
clf.fit(X, y)

selected_lasso = np.where(np.abs(clf.coef_[0]) > 1e-6)[0]
print(f" {len(selected_lasso)} features are selected by the Lasso:")
print(np.array(feature_names)[np.abs(clf.coef_[0]) > 1e-6])

#############################################################################
# Step 2: Evaluating the rejection set
# ------------------------------------------
#
# Since we do not have the ground truth, we cannot evaluate this selection set directly. To investigate the reliability of this method, we artificially increase the number of variables by adding synthetic noise features. These are completely irrelevant to the problem at hand.

# %%
repeats_noise = 3  # Number of synthetic noisy sets to add

X_to_shuff = X.copy()
noises = [X]

for k in range(repeats_noise):
    np.random.shuffle(X_to_shuff.T)
    noises.append(np.random.randn(n, p))

noisy = np.concatenate(noises, axis=1)
print("Shape after adding noise features:", noisy.shape)

# %% [markdown]
# There are 120 features -- 30 of them are real and 90 of them are fake and independent of the outcome. We now apply the Lasso to the augmented dataset:

# %%
clf_noisy = LogisticRegressionCV(penalty="l1", solver="liblinear", max_iter=int(1e4))
clf_noisy.fit(noisy, y)

selected_logl1 = np.where(np.abs(clf_noisy.coef_[0]) > 1e-6)[0]

# Count how many selected features are actually noise
num_false_discoveries = np.sum(selected_logl1 >= p)
print(f"The Lasso makes at least {num_false_discoveries} False Discoveries!!")

# %% [markdown]
# To mitigate this problem, we can use one of the statistically controlled variable selection methods implemented in hidimstat. This ensures that the proportion of False Discoveries is below a certain bound set by the user in all scenarios.

#############################################################################
# Step 3: Controlled variable selection with Knockoffs
# ----------------------------------------------------
#
# We use the Model-X Knockoff procedure to control the FDR (False Discovery Rate). To setup a fair comparison between the two methods, we first run the Knockoffs procedure on the original data and observe the results:

# %%
from hidimstat.knockoffs import model_x_knockoff

fdr = 0.25

selected, test_scores, threshold, X_tildes = model_x_knockoff(
    X,
    y,
    estimator=LogisticRegressionCV(
        solver="liblinear",
        penalty="l1",
    ),
    n_bootstraps=1,
    random_state=seed,
    preconfigure_estimator=None,
    fdr=fdr,
)

print("Selected features with Knockoffs:")
print(np.array(feature_names)[selected])


# %% [markdown]
# We now apply the Knockoffs procedure to the noisy dataset:

# %%
selected, test_scores, threshold, X_tildes = model_x_knockoff(
    X,
    y,
    estimator=LogisticRegressionCV(
        solver="liblinear",
        penalty="l1",
    ),
    n_bootstraps=1,
    random_state=seed,
    preconfigure_estimator=None,
    fdr=fdr,
)

print("Selected features with Knockoffs:")
print(np.array(feature_names)[selected])

# Count how many selected features are actually noise
num_false_discoveries = np.sum(selected >= p)
print(f"Knockoffs make at least {num_false_discoveries} False Discoveries")

# %% [markdown]
# Knockoffs select the same variable as previously and select **no fake variables**.
