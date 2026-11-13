.. hidimstat documentation master file, created by
   sphinx-quickstart on Fri April 23 12:22:52 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

HiDimStat: High-dimensional statistical inference tool for Python
=================================================================
|Build| |CircleCI/Documentation| |codecov| |codestyle|

.. |Build| image:: https://github.com/mind-inria/hidimstat/actions/workflows/ci.yml/badge.svg?branch=main
   :target: https://github.com/mind-inria/hidimstat/actions/workflows/ci.yml?branch=main

.. |CircleCI/Documentation| image:: https://circleci.com/gh/mind-inria/hidimstat.svg?style=shield
   :target: https://app.circleci.com/pipelines/github/mind-inria/hidimstat

.. |codecov| image:: https://codecov.io/github/mind-inria/hidimstat/branch/main/graph/badge.svg?token=O1YZDTFTNS
   :target: https://app.codecov.io/github/mind-inria/hidimstat

.. |codestyle| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black

The HiDimStat package implements different estimators that
compute variable importance for single or groups of features,
and perform variable selection with statistical guarantees.
The library provides statistical inference methods to solve the
problem of support recovery in the context of high-dimensional and
spatially structured data.


Documentation & Examples
------------------------

API documentation about the main HiDimStat functions is available
:ref:`here <api_documentation>`, and a :ref:`User guide <user_guide>` provides
further details on the various variable importance methods and concepts used in the library.

.. grid:: 1 2 3 3
   :gutter: 3

   .. grid-item-card:: Methods
      :link-type: ref
      :img-bottom: /generated/gallery/examples/images/thumb/sphx_glr_plot_cfi_thumb.png
      :img-alt: Variable Importance Method

      You can find examples for each Variable Importance Method:
      :ref:`LOCO <sphx_glr_generated_gallery_examples_plot_loco.py>`,
      :ref:`LOCI <sphx_glr_generated_gallery_examples_plot_loci.py>`,
      :ref:`PFI <sphx_glr_generated_gallery_examples_plot_pitfalls_permutation_importance.py>`,
      :ref:`CFI <sphx_glr_generated_gallery_examples_plot_cfi.py>`,
      :ref:`Desparsified Lasso <sphx_glr_generated_gallery_examples_plot_dl_diabetes_confidence.py>`,
      :ref:`DCRT <sphx_glr_generated_gallery_examples_plot_dcrt_logit.py>`,
      :ref:`Model-X Knockoffs <sphx_glr_generated_gallery_examples_plot_knockoffs_wisconsin.py>`,
      :ref:`SAGE <sphx_glr_generated_gallery_examples_plot_msage.py>`,
      :ref:`Cluster Importance <sphx_glr_generated_gallery_examples_plot_conditional_vs_marginal_xor_data.py>`,
      and :ref:`Ensemble Importance <sphx_glr_generated_gallery_examples_plot_conditional_vs_marginal_xor_data.py>`.


   .. grid-item-card:: Visualization
      :img-bottom: /generated/gallery/examples/images/thumb/sphx_glr_plot_partial_dependency_plot_thumb.png
      :img-alt: Visualization
      :link-type: ref

      The library implements two different feature importance visualization methods
      such as :ref:`Partial Dependency Plot <sphx_glr_generated_gallery_examples_plot_partial_dependency_plot.py>`
      and :ref:`Accumulated Local Effects <sphx_glr_generated_gallery_examples_plot_accumulated_local_effects.py>`


   .. grid-item-card:: Spatially structured data
      :img-bottom: /generated/gallery/examples/images/thumb/sphx_glr_plot_fmri_data_example_thumb.png
      :img-alt: Image stuff
      :link-type: ref

      You can find here applications to spatially structured data, such as
      :ref:`fMRI data <sphx_glr_generated_gallery_examples_plot_fmri_data_example.py>`,
      :ref:`somatosensory MEG data <sphx_glr_generated_gallery_examples_plot_meg_somato.py>`,
      or the :ref:`MNIST digit dataset <sphx_glr_generated_gallery_examples_plot_digits.py>`

All examples are available :ref:`there <general_examples>`.
If you have cloned the repository, you can run any of the examples locally by running the
following command in a terminal:

.. code-block::

  python examples/plot_2D_simulation_example.py

To run examples it is necessary to install the ``seaborn`` package.
Some additional dependencies might be required depending on the target example
to run. You can find further instructions on building the documentation
:ref:`here <developer_documentation_build>`.


Installation & Dependencies
---------------------------

HiDimStat works only with Python 3, ideally Python 3.10+. To install,
run the following from a terminal in your Python environment::

  pip install hidimstat

Or if you want the latest version available, to contribute to
the development of this project for instance::

  git clone https://github.com/mind-inria/hidimstat.git
  cd hidimstat
  pip install -e .

HiDimStat depends on the following packages::

  joblib >= 1.4
  numpy >= 1.19.5
  pandas >= 2.2
  scipy >= 1.13
  scikit-learn >= 1.6
  tqdm >= 4.66.3


References
----------

If you use our package, we would appreciate citations to the relevant
aforementioned papers. You can also contact us by opening an issue on the
`Github repository <https://github.com/mind-inria/hidimstat>`_ to share
your work which we would love helping to promote it.


.. toctree::
  :hidden:
  :maxdepth: 1

  api
  user_guide
  generated/gallery/examples/index
  glossary_and_notations
  whats_news/whats_news
  dev/index
