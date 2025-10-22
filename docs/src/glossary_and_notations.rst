.. _glossary_and_notations:


===========================
Glossary and Notations
===========================

Notations
---------

.. glossary::

    minus index
    :math:`X^{-j}`
        The minus index notation is used to denote all features except the one with the 
        given index. For instance, :math:`X_{-j}` denotes all features except the 
        :math:`j^{th}` one.

    minus group
    :math:`X_{-G}`
        Similar to the :term:`minus index` notation for individual features, we use the minus 
        index notation to denote the complement of a group of features. For instance, 
        :math:`X_{-G}` denotes all features except the ones in the group :math:`G`.


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
            \text{FDP} = \frac{|\hat S \cap \hat S \setminus S^*|}{|\hat S|}.

        where :math:`|\cdot|` denotes the cardinality of a set.

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