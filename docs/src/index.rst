.. hidimstat documentation master file, created by
   sphinx-quickstart on Fri April 23 12:22:52 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

HiDimStat: High-dimensional statistical inference tool for Python
=================================================================
|Build Status| |codecov|

The HiDimStat package provides statistical inference methods to solve the
problem of support recovery in the context of high-dimensional and
spatially structured data.


Installation
------------

HiDimStat working only with Python 3, ideally Python 3.6+. For installation,
run the following from terminal::

  pip install hidimstat

Or if you want the latest version available (for example to contribute to
the development of this project)::

  git clone https://github.com/ja-che/hidimstat.git
  cd hidimstat
  pip install -e .


Dependencies
------------

HiDimStat depends on the following packages::

  joblib
  numpy
  scipy
  scikit-learn


To run examples it is neccessary to install ``matplotlib``, and to run tests it
is also needed to install ``pytest``.


Documentation & Examples
------------------------

Documentation about the main HiDimStat functions is available
`here <api.html>`_ and examples are available `here <auto_examples/index.html>`_.

As of now, there are three different examples (Python scripts) that
illustrate how to use the main HiDimStat functions.
In each example we handle a different kind of dataset:
``plot_2D_simulation_example.py`` handles a simulated dataset with a 2D
spatial structure,
``plot_fmri_data_example.py`` solves the decoding problem on Haxby fMRI dataset,
``plot_meg_data_example.py`` tackles the source localization problem on several
MEG/EEG datasets.

.. code-block::

  # For example run the following command in terminal
  python plot_2D_simulation_example.py


Build the documentation
-----------------------

To build the documentation you will need to run:

.. code-block::

    pip install -U sphinx_gallery sphinx_bootstrap_theme
    cd doc
    make html


References
----------

The algorithms developed in this package have been detailed in several
conference/journal articles that can be downloaded at
`https://ja-che.github.io/ <https://ja-che.github.io/research.html>`_.

Main references
~~~~~~~~~~~~~~~

Ensemble of Clustered desparsified Lasso (ECDL):

* Chevalier, J. A., Salmon, J., & Thirion, B. (2018). Statistical inference
  with ensemble of clustered desparsified lasso. In International Conference
  on Medical Image Computing and Computer-Assisted Intervention
  (pp. 638-646). Springer, Cham.

* Chevalier, J. A., Nguyen, T. B., Thirion, B., & Salmon, J. (2021).
  Spatially relaxed inference on high-dimensional linear models.
  arXiv preprint arXiv:2106.02590.

Aggregation of multiple Knockoffs (AKO):

* Nguyen T.-B., Chevalier J.-A., Thirion B., & Arlot S. (2020). Aggregation
  of Multiple Knockoffs. In Proceedings of the 37th International Conference on
  Machine Learning, Vienna, Austria, PMLR 119.

Application to decoding (fMRI data):

* Chevalier, J. A., Nguyen T.-B., Salmon, J., Varoquaux, G. & Thirion, B.
  (2021). Decoding with confidence: Statistical control on decoder maps.
  In NeuroImage, 234, 117921.

Application to source localization (MEG/EEG data):

* Chevalier, J. A., Gramfort, A., Salmon, J., & Thirion, B. (2020).
  Statistical control for spatio-temporal MEG/EEG source imaging with
  desparsified multi-task Lasso. In Proceedings of the 34th Conference on
  Neural Information Processing Systems (NeurIPS 2020), Vancouver, Canada.

If you use our packages, we would appreciate citations to the relevant
aforementioned papers.

Other useful references
~~~~~~~~~~~~~~~~~~~~~~~

For de-sparsified(or de-biased) Lasso:

* Javanmard, A., & Montanari, A. (2014). Confidence intervals and hypothesis
  testing for high-dimensional regression. The Journal of Machine Learning
  Research, 15(1), 2869-2909.

* Zhang, C. H., & Zhang, S. S. (2014). Confidence intervals for low dimensional
  parameters in high dimensional linear models. Journal of the Royal
  Statistical Society: Series B: Statistical Methodology, 217-242.

* Van de Geer, S., Bühlmann, P., Ritov, Y. A., & Dezeure, R. (2014). On
  asymptotically optimal confidence regions and tests for high-dimensional
  models. The Annals of Statistics, 42(3), 1166-1202.

For Knockoffs Inference:

* Barber, R. F; Candès, E. J. (2015). Controlling the false discovery rate
  via knockoffs. Annals of Statistics. 43 , no. 5,
  2055--2085. doi:10.1214/15-AOS1337. https://projecteuclid.org/euclid.aos/1438606853

* Candès, E., Fan, Y., Janson, L., & Lv, J. (2018). Panning for gold: Model-X
  knockoffs for high dimensional controlled variable selection. Journal of the
  Royal Statistical Society Series B, 80(3), 551-577.

.. |Build Status| image:: https://travis-ci.com/ja-che/hidimstat.svg?branch=main
   :target: https://codecov.io/gh/ja-che/hidimstat

.. |codecov| image:: https://codecov.io/gh/ja-che/hidimstat/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/ja-che/hidimstat


API
---

.. toctree::
    :maxdepth: 1

    api.rst
