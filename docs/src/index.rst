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
   :target: https://circleci.com/gh/mind-inria/hidimstat

.. |codecov| image:: https://codecov.io/github/mind-inria/hidimstat/branch/main/graph/badge.svg?token=O1YZDTFTNS
   :target: https://codecov.io/github/mind-inria/hidimstat

.. |codestyle| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black

The HiDimStat package provides statistical inference methods to solve the
problem of support recovery in the context of high-dimensional and
spatially structured data. The library implements different estimators that
compute variable importance for single or groups of features,
and perform variable selection with statistical guarantees.


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


Documentation & Examples
------------------------

Documentation about the main HiDimStat functions is available
:ref:`here <api_documentation>`
and examples are available :ref:`there <general_examples>`.

The examples cover different aspects:

* Presentations of a variable importance estimator, how they work, and their limitations,
  such as with :ref:`sphx_glr_generated_gallery_examples_plot_cfi.py`,
  or :ref:`sphx_glr_generated_gallery_examples_plot_msage.py`.
* Applications on spatially structured data, be it with
  :ref:`sphx_glr_generated_gallery_examples_plot_2D_simulation_example.py`,
  or for :ref:`sphx_glr_generated_gallery_examples_plot_fmri_data_example.py`,
  or even :ref:`sphx_glr_generated_gallery_examples_plot_meg_somato.py`.


Many more examples can be found in the example gallery.
If you have cloned the repository, you can run any of the examples locally by running the
following command in a terminal:

.. code-block::

  python examples/plot_2D_simulation_example.py

To run examples it is necessary to install the ``seaborn`` package.
Some additional dependencies might be required depending on the target example
to run. You can find further instructions on building the documentation
:ref:`here <developer_documentation_build>`.


References
----------

The algorithms developed in this package have been detailed in several
conference/journal articles that can be downloaded at
`https://team.inria.fr/mind/publications/ <https://team.inria.fr/mind/publications/>`_.

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
