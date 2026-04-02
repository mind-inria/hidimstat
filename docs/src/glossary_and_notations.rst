.. _glossary_and_notations:


===========================
Glossary and Notations
===========================

Notations
---------

.. glossary::

    target
    :math:`Y`, ``y``
        In the documentation, the target variable is denoted by a capital letter
        :math:`Y` when referring to it as a random variable. In the API and code
        examples, the target variable is denoted by a lowercase letter ``y``, following
        the ``scikit-learn`` convention.

    index
    :math:`X^{j}`
        The superscript index notation is used to denote the :math:`j^{th}` feature of
        the feature vector :math:`X`.

    minus index
    :math:`X^{-j}`
        The minus index notation is used to denote all features except the one with the
        given index. For instance, :math:`X^{-j}` denotes all features except the
        :math:`j^{th}` one.

    minus group
    :math:`X^{-G}`
        Similar to the :term:`minus index` notation for individual features, we use the minus
        index notation to denote the complement of a group of features. For instance,
        :math:`X^{-G}` denotes all features except the ones in the group :math:`G`.

    marginal permutation of a feature
    :math:`X^{\pi (j)}`
        To denote a perturbed version of an input :math:`X` where the :math:`j^{th}`
        feature has been permuted, we use the superscript notation :math:`\pi (j)`. This
        permutation is marginal, meaning that the values of the :math:`j^{th}` feature
        are shuffled across samples, independently from the other features.

    conditional permutation of a feature
    :math:`X^{\pi(j| -j)}`
        To denote a perturbed version of an input :math:`X` where the :math:`j^{th}`
        feature has been sampled from its conditional distribution given all other
        features, we use the notation :math:`X^{\pi(j| -j)}`. This means that the
        values of the :math:`j^{th}` are drawn from the distribution
        :math:`P(X^{j} | X^{-j})`.

    knockoff feature
    :math:`\tilde X`
        The knockoff input :math:`\tilde X` is a perturbed version of the original
        input :math:`X` constructed such that each knockoff feature is pairwise
        exchangeable with the original feature and independent from the output :math:`Y`
        conditionally on the original features.


Glossary
--------

.. glossary::

    CFI
    Conditional Feature Importance
        Conditional Feature Importance (CFI) is a measure of feature importance that
        consists in sampling a feature of interest from the conditional distribution
        of that feature given all other features, and measuring the performance drop
        triggered by this perturbation.

    CluDL
    Clustered Desparsified Lasso
        Clustered Desparsified Lasso (CluDL) is an extension of the Desparsified
        Lasso to clusters of features. It aims at overcoming the limitations of
        the Desparsified Lasso when the number of features is large and the
        correlations between them are strong.

    D0CRT
    Distilled Conditional Randomization Testing
        Distilled Conditional Randomization Testing (dCRT) is a method for feature
        selection based on conditional independence testing, which uses a distillation
        step to reduce computational cost.

    EnCluDL
    Ensemble Clustered Desparsified Lasso
        Ensemble Clustered Desparsified Lasso (EnCluDL) is an extension of the
        Clustered Desparsified Lasso that combines multiple clusterings to de-randomize
        the procedure and improve robustness.

    FDP
    False Discovery Proportion
        The False Discovery Proportion (FDP) is the ratio between the number of false
        discoveries and the total number of discoveries. Denoting :math:`\hat S` the
        estimated set of important features, and :math:`S^*` the true set of important
        features, the FDP is defined as:

        .. math::
            \text{FDP}(\hat S) = \frac{|\hat S \setminus S^*|}{\max(|\hat S|, 1)}.

        where :math:`|\cdot|` denotes the cardinality of a set.

    FDR
    False Discovery Rate
        The False Discovery Rate (FDR) is the expected value of the False Discovery
        Proportion (FDP). For a selection set :math:`\hat S` it is defined as
        :math:`\text{FDR}(\hat S) = \mathbb{E}[\text{FDP}(\hat S)]`.

    LOCO
    Leave-One-Covariate-Out
        The Leave-One-Covariate-Out (LOCO) is a measure of feature importance that
        consists in retraining a predictive model without the feature of interest and
        measuring performance drop triggered by this ablation.

    PFI
    Permutation Feature Importance
        The Permutation Feature Importance (PFI) is a measure of feature importance
        that consists in permuting the values of the feature of interest and measuring
        the performance drop triggered by this perturbation.
