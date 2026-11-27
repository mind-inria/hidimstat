.. # Add the reference for the badges
.. ## Reference to the CI status
.. |Linter&Tests| image:: https://github.com/mind-inria/hidimstat/actions/workflows/ci.yml/badge.svg?branch=main
   :target: https://github.com/mind-inria/hidimstat/actions/workflows/ci.yml?query=branch%3Amain
.. |CircleCI/Documentation| image:: https://circleci.com/gh/mind-inria/hidimstat.svg?style=shield
   :target: https://circleci.com/gh/mind-inria/hidimstat?branch=main
.. |CodeCov| image:: https://codecov.io/github/mind-inria/hidimstat/branch/main/graph/badge.svg?token=O1YZDTFTNS
   :target: https://codecov.io/github/mind-inria/hidimstat
.. ## Distribution python
.. |PyPi| image:: https://img.shields.io/pypi/v/hidimstat.svg
   :target: https://pypi.org/project/hidimstat/
.. |PyPi_download| image:: https://img.shields.io/pypi/dm/hidimstat
   :target: https://pypi.org/project/hidimstat/
.. |PythonVersion| image:: https://img.shields.io/pypi/pyversions/hidimstat.svg?color=informational
   :target: https://pypi.org/project/hidimstat/
.. |Latest release| image:: https://img.shields.io/github/release/mind-inria/hidimstat.svg?color=brightgreen&label=latest%20release
.. ## Additional badge
.. |License| image:: https://img.shields.io/github/license/mind-inria/hidimstat
   :target: https://opensource.org/license/bsd-3-clause
.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black

.. # Add minimal dependency of the main packages
  ## This need to update in same time that pyproject.toml
.. |PythonMinVersion| replace:: 3.10
.. |JoblibMinVersion| replace:: 1.2
.. |NumPyMinVersion| replace:: 1.25
.. |PandasMinVersion| replace:: 2.0
.. |SklearnMinVersion| replace:: 1.4
.. |SciPyMinVersion| replace:: 1.6
.. ## for plotting and for examples
.. |MatplotlibMinVersion| replace:: 3.9.0
.. |SeabornMinVersion| replace:: 0.9.0

=================================================================
HiDimStat: High-dimensional statistical inference tool for Python
=================================================================

.. # Add the different badge

|Linter&Tests| |CircleCI/Documentation| |CodeCov| |Black|

|PyPi| |PyPi_download| |PythonVersion| |Latest release|

|License|

.. # Short description of the library

The HiDimStat package implements different estimators that
compute variable importance for single or groups of features,
and perform variable selection with statistical guarantees.
The library provides statistical inference methods to solve the
problem of support recovery in the context of high-dimensional and
spatially structured data.

.. # Add useful links

If you like the package, spread the word and ⭐ our `official repository
<https://github.com/mind-inria/hidimstat>`_!

Visit our website, https://hidimstat.github.io/, for more information.

..
  ## TODO: Add short citation when this will be ready
  If you use HiDimStat for your published research, we kindly ask you to :ref:`cite<citation>` our article:
  short reference

If you have any problems, please report them to the `GitHub issue tracker <https://github.com/mind-inria/hidimstat/issues>`_
or contribute to the library by opening a pull request.

Main Features
-------------

Variable Importance Methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The library features different Variable Importance Methods (VIM):

- `LOCO <https://hidimstat.github.io/stable/generated/gallery/examples/plot_loco.html>`_:
  A refit-based model agnostic approach that measures how much a model's
  predictive performance changes when a specific feature is removed while
  conditioning on the remaining features.
- `LOCI <https://hidimstat.github.io/stable/generated/gallery/examples/plot_loci.html>`_:
  A refit-based model agnostic approach that compares the predictive performance
  of the empty model (dummy, predicting at change level) with the performance of
  a model refitted just with the feature(s) of interest.
- `PFI <https://hidimstat.github.io/stable/generated/gallery/examples/plot_pitfalls_permutation_importance.html>`_:
  A perturbation-based method that compares the predictive performance of a model on
  unmodified test data to its performance when the studied feature is marginally permutated.
- `CFI <https://hidimstat.github.io/stable/generated/gallery/examples/plot_cfi.html>`_:
  A perturbation-based method that compares the predictive performance of a model on
  unmodified test data to its performance when the studied feature is conditionally permutated.
- `Desparsified Lasso <https://hidimstat.github.io/stable/generated/gallery/examples/plot_dl_diabetes_confidence.html>`_:
  Also known as debiased Lasso, it is a method that aims at estimating
  the regression coefficients.
- `DCRT <https://hidimstat.github.io/stable/generated/gallery/examples/plot_dcrt_logit.html>`_:
  A variable selection method that tests whether each feature is conditionally
  independent of the target given all other features.
- `Model-X Knockoffs <https://hidimstat.github.io/stable/generated/gallery/examples/plot_knockoffs_wisconsin.html>`_:
  A variable selection method that controls the false discovery rate (FDR) by
  producing synthetic 'knockoff' variables that mimic the correlation structure
  of features without depending on the outcome.
- `Shapley Additive Global Explanation (SAGE) <https://hidimstat.github.io/stable/generated/gallery/examples/plot_msage.html>`_:
  Built on top of `Shapley values <https://en.wikipedia.org/wiki/Shapley_value>`_,
  it determines each feature's contribution by considering how
  much the overall model performance changes when they join each possible combination of
  other features, and then averaging those changes.
- `Cluster Importance <https://hidimstat.github.io/stable/generated/gallery/examples/plot_conditional_vs_marginal_xor_data.html>`_:
  A meta VIM that produces clusters of features before passing them to a VIM.
- `Ensemble Importance <https://hidimstat.github.io/stable/generated/gallery/examples/plot_conditional_vs_marginal_xor_data.html>`_:
  A meta VIM that bootstraps a VIM.

Visualization
^^^^^^^^^^^^^
The library currently implements two visualization methods:

- `Partial Dependency Plot <https://hidimstat.github.io/stable/generated/gallery/examples/plot_partial_dependency_plot.html>`_:
  A plot to visualize the marginal effect of one or two features on a model's
  prediction.
- `Accumulated Local Effects <https://hidimstat.github.io/stable/generated/gallery/examples/plot_accumulated_local_effects.html>`_:
  A plot that describes how features influence the prediction of a model on by
  averaging local effects.

User Guide
^^^^^^^^^^

You can find additional information about the different methods in our `User guide <https://hidimstat.github.io/stable/user_guide.html>`_,
or explore the possibilities of the library through the `Examples Gallery <https://hidimstat.github.io/stable/generated/gallery/examples/index.html>`_.

Installation
------------

Dependencies
^^^^^^^^^^^^

.. # Add dependency of the project
  TODO Need to match with pyproject.toml

HiDimStat requires:

- Python (>= |PythonMinVersion|)
- joblib (>= |JoblibMinVersion|)
- Pandas (>= |PandasMinVersion|)
- Scikit-learn (>= |SklearnMinVersion|, < 1.9)
- SciPy (>= |SciPyMinVersion|)

HiDimStat's plotting capabilities require Matplotlib (>= |MatplotlibMinVersion|).

To run the examples, Matplotlib (>= |MatplotlibMinVersion|) and seaborn (>=
|SeabornMinVersion|) are required.

User installation
^^^^^^^^^^^^^^^^^

.. # Add the instruction for installation
  TODO add conda when it will be accessible

HiDimStat can easily be installed via ``pip``.
For more installation information,
see the `installation instructions <https://hidimstat.github.io/dev/index.html#installation>`_.

.. code-block:: bash

  pip install -U hidimstat

Contribute
----------

.. # Add short description for contribution to the library

The best way to support the development of HiDimStat is to spread the word!

HiDimStat aims to be supported by an active community, and we welcome
contributions to our code and documentation.

For bug reports, feature requests, documentation improvements, or other issues,
you can create a `GitHub issue <https://github.com/mind-inria/hidimstat/issues>`_.

If you want to contribute directly to the library, check the
`how to contribute <https://hidimstat.github.io/stable/dev/how_to_contribute.html>`_ page
on the website for more information.

Contact us
----------

.. # Add a way to contact maintainers
  TODO this needs to be updated when there is a change of maintainers

| Currently, this library is supported by the `INRIA <https://www.inria.fr/en>`_
  team `MIND <https://www.inria.fr/fr/mind>`_.
| If you want to report a problem or suggest an enhancement, we would love for you
  to `open an issue <https://github.com/mind-inria/hidimstat/issues/new>`_ at
  this GitHub repository so we can address it quickly.
| For less formal discussions or to exchange ideas, you can contact the main
  contributors:

+-------------------+-------------------+-------------------+-------------------+-------------------+
|   Lionel Kusch    | Bertrand Thirion  |  Joseph Paillard  | Angel Reyero Lobo |   Marc Hulcelle   |
+-------------------+-------------------+-------------------+-------------------+-------------------+
|    |avatar LK|    |    |avatar BT|    |     |avatar JP|   |    |avatar AR|    |    |avatar MH|    |
+-------------------+-------------------+-------------------+-------------------+-------------------+

.. |avatar LK| image:: https://avatars.githubusercontent.com/u/17182418?v=4
  :target: https://github.com/lionelkusch
.. |avatar BT| image:: https://avatars.githubusercontent.com/u/234454?v=4
  :target: https://github.com/bthirion
.. |avatar JP| image:: https://avatars.githubusercontent.com/u/56166877?v=4
  :target: https://github.com/jpaillard
.. |avatar AR| image:: https://avatars.githubusercontent.com/u/79699169?v=4
  :target: https://github.com/AngelReyero
.. |avatar MH| image:: https://avatars.githubusercontent.com/u/9165180?v=4
  :target: https://github.com/GrituX

Citation
--------

If you use a HiDimStat method for your research, you'll find the associated
reference paper in the method description, and we recommend that you cite it.

..
  TODO add the section for citing the library once a Zenodo repository is made
  or a paper is published.

If you publish a paper using HiDimStat, please contact us or open an issue!
We would love to hear about your work and help you promote it.

Acknowledgments
---------------

This project has been funded by `Labex DigiCosme <https://anr.fr/ProjetIA-11-LABX-0045>`_
(ANR-11-LABEX-0045-DIGICOSME) as part of the program
`Investissement d’Avenir <https://anr.fr/ProjetIA-11-IDEX-0003>`_
(ANR-11-IDEX-0003-02), by the `Fast Big project <https://anr.fr/Projet-ANR-17-CE23-0011>`_
(ANR-17-CE23-0011), by the `KARAIB AI Chair <https://anr.fr/Projet-ANR-20-CHIA-0025>`_
(ANR-20-CHIA-0025-01), and by the `VITE project <https://anr.fr/Projet-ANR-23-CE23-0016>`_ (ANR-23-CE23-0016).
This study has also been supported by the European Union’s Horizon 2020 research and innovation program
as part of the program `Human Brain Project SGA3 <https://cordis.europa.eu/project/id/945539>`_
(Grant Agreement No. 945539) and `EBRAIN-Health <https://cordis.europa.eu/project/id/101058516>`_
(Grant Agreement No. 101058516).
