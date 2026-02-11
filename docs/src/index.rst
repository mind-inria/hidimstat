HiDimStat: High-dimensional statistical inference tool for Python
=================================================================
|Build| |CircleCI/Documentation| |codecov| |codestyle|

The HiDimStat package provides statistical inference methods to solve the
problem of support recovery in the context of high-dimensional and
spatially structured data.


Installation
------------

HiDimStat working only with Python 3, ideally Python 3.10+. For installation,
run the following from terminal::

  pip install hidimstat

Or if you want the latest version available (for example to contribute to
the development of this project)::

  git clone https://github.com/mind-inria/hidimstat.git
  cd hidimstat
  pip install -e .


Dependencies
------------

HiDimStat depends on the following packages::

  joblib
  numpy
  pandas
  scipy
  scikit-learn
  tqdm

To run examples it is necessary to install ``seaborn``, and to run tests it
is also needed to install ``pytest``.


Documentation & Examples
------------------------

Documentation about the main HiDimStat functions is available
:ref:`here <api_documentation>`
and examples are available
:ref:`there <general_examples>`.

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

    pip install -U '.[doc]'
    cd docs
    make html


References
----------

The algorithms developed in this package have been detailed in several
conference/journal articles that can be downloaded at
`https://team.inria.fr/mind/publications/ <https://team.inria.fr/mind/publications/>`_.

Main references
~~~~~~~~~~~~~~~

* Ensemble of Clustered desparsified Lasso (ECDL): :cite:t:`chevalier2018statistical`, :cite:t:`chevalier2022spatially`

* Aggregation of multiple Knockoffs (AKO): :cite:t:`pmlr-v119-nguyen20a`

* Application to decoding (fMRI data): :cite:t:`chevalier2021decoding`

* Application to source localization (MEG/EEG data): :cite:t:`chevalier2020statistical`

* Single/Group statistically validated importance using conditional permutations: :cite:t:`Chamma_NeurIPS2023`, :cite:t:`Chamma_AAAI2024`

If you use our packages, we would appreciate citations to the relevant
aforementioned papers.

Other useful references
~~~~~~~~~~~~~~~~~~~~~~~

* For de-sparsified (or de-biased) Lasso: :cite:t:`javanmard2014confidence`, :cite:p:`zhang2014confidence`, :cite:t:`van2014asymptotically`

* For Knockoffs Inference: :cite:t:`barber2015controlling`, :cite:t:`candes2018panning`

.. |Build| image:: https://github.com/mind-inria/hidimstat/actions/workflows/ci.yml/badge.svg?branch=main
   :target: https://github.com/mind-inria/hidimstat/actions/workflows/ci.yml?branch=main

.. |CircleCI/Documentation| image:: https://circleci.com/gh/mind-inria/hidimstat.svg?style=shield
   :target: https://circleci.com/gh/mind-inria/hidimstat

.. |codecov| image:: https://codecov.io/github/mind-inria/hidimstat/branch/main/graph/badge.svg?token=O1YZDTFTNS
   :target: https://codecov.io/github/mind-inria/hidimstat

.. |codestyle| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black


References
----------
.. bibliography:: ../tools/references.bib


.. toctree::
  :hidden:
  :maxdepth: 1

  api
  user_guide
  generated/gallery/examples/index
  glossary_and_notations
  whats_news/whats_news
  dev/index
