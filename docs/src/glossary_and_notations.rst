.. _glossary_and_notations:


===========================
Glossary and Notations
===========================

Notations
---------

.. glossary::

    minus index
    individual features
        The minus index notation is used to denote all features except the one with the 
        given index. For instance, :math:`X_{-j}` denotes all features except the 
        :math:`j^{th}` one.

    Feature groups
        Similar to the minus index notation for individual features, we use the minus 
        index notation to denote the complement of a group of features. For instance, 
        :math:`X_{-G}` denotes all features except the ones in the group :math:`G`.


Glossary
--------

.. glossary::
    
    FDP
    False Discovery Proportion
        The False Discovery Proportion (FDP) is the ratio between the number of false
        discoveries and the total number of discoveries. Denoting :math:`\hat S` the 
        estimated set of important features, and :math:`S^*` the true set of important 
        features, the FDP is defined as:

        .. math::
            \text{FDP} = \frac{|\hat S \cap \hat S \ S^*|}{|\hat S|}.

        where :math:`|\cdot|` denotes the cardinality of a set, and using the convention 
        that :math:`\text{FDP} = 0` if :math:`\hat S = \emptyset`
        
    e-values
        TODO