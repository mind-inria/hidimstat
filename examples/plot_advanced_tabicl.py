import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils import resample
from tabicl import TabICLClassifier

from hidimstat import D0CRT

dataset = fetch_openml("adult", version=2, as_frame=True)
X, y, feat_names = dataset.data, dataset.target, dataset.feature_names
y = (y == ">50K").astype(int)

max_samples = 10000
X, y = resample(
    X,
    y,
    n_samples=max_samples,
    replace=False,
    random_state=0,
    stratify=y,
)

num_cols = X.select_dtypes(include=np.number).columns
cat_cols = X.select_dtypes(exclude=np.number).columns

preprocess = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            num_cols,
        ),
        (
            "cat",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]
            ),
            cat_cols,
        ),
    ],
)


X_proc = preprocess.fit_transform(X).toarray()
feat_names_proc = preprocess.get_feature_names_out()

"""rng = np.random.default_rng(0)

n, p = X_proc.shape
n_dup = 10

synthetic = []

for i in range(n_dup):
    base_idx = rng.integers(0, p, size=20)

    block = (
        0.95 * X_proc[:, base_idx]
        + 0.05 * rng.standard_normal((n, 20))
    )

    synthetic.append(block)

X_aug = np.hstack([X_proc] + synthetic)"""
X_train, X_test, y_train, y_test = train_test_split(
    X_proc, y, test_size=0.5, random_state=0
)
tabicl = TabICLClassifier(device="cpu")

tabicl.fit(X_train, y_train)
"""skorch_model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    max_iter=50,
    random_state=0,
)

skorch_model.fit(X_train, y_train)
skorch_pred = skorch_model.predict(X_test)"""
d0crt_lasso = D0CRT(
    estimator=LassoCV(random_state=0, n_jobs=1),
    screening_threshold=None,
    random_state=0,
)
d0crt_lasso.fit(
    X_train,
    y_train,
)
importances = d0crt_lasso.importance(X_test, y_test)
selection = d0crt_lasso.fdr_selection(fdr=0.2)

df = pd.DataFrame(
    {
        "feature": feat_names_proc,
        "importance": importances,
        "selected": selection,
    }
).sort_values("importance", ascending=False)

import matplotlib.pyplot as plt
import seaborn as sns

ax = sns.barplot(
    data=df.head(20),
    y="feature",
    x="importance",
    hue="selected",
    palette="muted",
    orient="h",
)
sns.despine()
ax.set_yticklabels(
    " ".join(x.get_text().split("__")[1:]) for x in ax.get_yticklabels()
)
plt.show()
