*****************************************************************
HiDimStat: High-dimensional statistical inference tool for Python
*****************************************************************
..
  Add the different badge

|Linter&Tests| |CircleCI/Documentation| |CodeCov|

|PyPi| |PyPi format| |PyPi_download| |Latest release| |PythonVersion|

|GitHub contributors| |GitHub commit activity| |License| |Black|

..
  Reference to the CI status
.. |Linter&Tests| image:: https://github.com/mind-inria/hidimstat/actions/workflows/main_workflow.yml/badge.svg?branch=main
    :target: https://github.com/mind-inria/hidimstat/actions/workflows/main_workflow.yml?query=branch%3Amain
.. |CircleCI/Documentation| image:: https://circleci.com/gh/mind-inria/hidimstat.svg?style=shield
    :target: https://circleci.com/gh/mind-inria/hidimstat?branch=main
.. |CodeCov| image:: https://codecov.io/github/mind-inria/hidimstat/branch/main/graph/badge.svg?token=O1YZDTFTNS
    :target: https://codecov.io/github/mind-inria/hidimstat
..
  Distribution python
.. |PyPi| image:: https://img.shields.io/pypi/v/hidimstat.svg
    :target: https://pypi.org/project/hidimstat/
.. |PyPi_download| image:: https://img.shields.io/pypi/dm/hidimstat
    :target: https://pypi.org/project/hidimstat/
.. |PyPi format| image:: https://img.shields.io/pypi/format/hidimstat
    :target: https://pypi.org/project/hidimstat/
.. |PythonVersion| image:: https://img.shields.io/pypi/pyversions/hidimstat.svg?color=informational
    :target: https://pypi.org/project/hidimstat/
.. |Latest release| image:: https://img.shields.io/github/release/mind-inria/hidimstat.svg?color=brightgreen&label=latest%20release
  :target: https://github.com/mind-inria/hidimstat/releases
..
  Additional badge
.. |GitHub contributors| image:: https://img.shields.io/github/contributors/mind-inria/hidimstat.svg?logo=github
  :target: https://github.com/mind-inria/hidimstat
.. |GitHub commit activity| image:: https://img.shields.io/github/commit-activity/y/mind-inria/hidimstat.svg?logo=github&color=%23ff6633
  :target: https://github.com/mind-inria/hidimstat
.. |License| image:: https://img.shields.io/github/license/mind-inria/hidimstat
    :target: https://opensource.org/license/bsd-3-clause
.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black


The HiDimStat package provides statistical inference methods to solve the problem
of support recovery in the context of high-dimensional and spatially structured data.

If you like the package, spread the word and ⭐ our `official repository  <https://github.com/mind-inria/hidimstat>`_!

Visit your website, https://hidimstat.github.io/, for more information.

..
  Add short citation when this will be ready
  If you use HiDimStat for your published research, we kindly ask you to :ref:`cite<citation>` our article:
  short reference

Find the important variables in your data using 
`our examples. <https://hidimstat.github.io/dev/auto_examples/index.html>`_ .

If you have any problems, please report them to the github issue tracker 
(https://github.com/mind-inria/hidimstat/issues) or contribute to the library 
by opening a Pull Request.



Installation
============

We recommend using HiDimStat with Python 3.12. For installation, we recommend 
using ``conda`` for python environment management. You can do so by running the 
following commands from the terminal::
  
  pip install hidimstat


Or if you want the latest version available 
(for example to contribute to the development of this project)::

  pip install -U git+https://github.com/mind-inria/hidimstat.git

or::

  git clone https://github.com/mind-inria/hidimstat.git
  cd hidimstat
  pip install -e .


Dependencies
============

- joblib
- numpy
- panda
- scipy
- scikit-learn

To run examples it is neccessary to install `matplotlib`, and to run tests it
is also needed to install ``pytest``.

Documentation & Examples
========================

All the documentation of HiDimStat is available at
 https://mind-inria.github.io/hidimstat/.

As of now in the ```examples``` folder there are three Python scripts that 
illustrate how to use the main HiDimStat functions. In each script we handle 
a different kind of dataset:

* ``plot_2D_simulation_example.py`` handles a simulated dataset with
  a 2D spatial structure,
* ``plot_fmri_data_example.py`` solves the decoding problem on 
  Haxby fMRI dataset, MEG/EEG datasets.

For example run the following command in terminal::
  python plot_2D_simulation_example.py

References
==========

The algorithms developed in this package have been detailed in several 
conference/journal articles that can be downloaded at 
https://mind-inria.github.io/research.html.

Main references
^^^^^^^^^^^^^^^

Ensemble of Clustered desparsified Lasso (ECDL):

* Chevalier, J. A., Salmon, J., & Thirion, B. (2018). **Statistical inference
  with ensemble of clustered desparsified lasso**. In *International Conference
  on Medical Image Computing and Computer-Assisted Interventio* (pp. 638-646). 
  Springer, Cham.

* Chevalier, J. A., Nguyen, T. B., Thirion, B., & Salmon, J. (2021). **Spatially
  relaxed inference on high-dimensional linear models**. arXiv preprint arXiv:2106.02590.

Aggregation of multiple Knockoffs (AKO):

* Nguyen T.-B., Chevalier J.-A., Thirion B., & Arlot S. (2020). **Aggregation
  of Multiple Knockoffs**. In *Proceedings of the 37th International Conference on
  Machine Learning*, Vienna, Austria, PMLR 119.

Application to decoding (fMRI data):

* Chevalier, J. A., Nguyen T.-B., Salmon, J., Varoquaux, G. & Thirion, B. (2021).
  **Decoding with confidence: Statistical control on decoder maps**. 
  In *NeuroImage*, 234, 117921.

Application to source localization (MEG/EEG data):

* Chevalier, J. A., Gramfort, A., Salmon, J., & Thirion, B. (2020). 
  **Statistical control for spatio-temporal MEG/EEG source imaging with
  desparsified multi-task Lasso**. In *Proceedings of the 34th Conference on
  Neural Information Processing Systems (NeurIPS 2020)*, Vancouver, Canada.

Single/Group statistically validated importance using conditional permutations:

* Chamma, A., Thirion, B., & Engemann, D. (2024). **Variable importance in 
  high-dimensional settings requires grouping**. In *Proceedings of the 38th 
  Conference of the Association for the Advancement of Artificial 
  Intelligence(AAAI 2024)*, Vancouver, Canada.

* Chamma, A., Engemann, D., & Thirion, B. (2023). **Statistically Valid Variable
  Importance Assessment through Conditional Permutations**. In *Proceedings of
  the 37th Conference on Neural Information Processing Systems (NeurIPS 2023)*, 
  New Orleans, USA.

If you use our packages, we would appreciate citations to the relevant 
aforementioned papers.

Other useful references:
^^^^^^^^^^^^^^^^^^^^^^^^

For de-sparsified(or de-biased) Lasso:

* Javanmard, A., & Montanari, A. (2014). **Confidence intervals and hypothesis
  testing for high-dimensional regression**. *The Journal of Machine Learning
  Research*, 15(1), 2869-2909.

* Zhang, C. H., & Zhang, S. S. (2014). **Confidence intervals for low dimensional
  parameters in high dimensional linear models**. *Journal of the Royal
  Statistical Society: Series B: Statistical Methodology*, 217-242.

* Van de Geer, S., Bühlmann, P., Ritov, Y. A., & Dezeure, R. (2014). **On
  asymptotically optimal confidence regions and tests for high-dimensional
  models**. *The Annals of Statistics*, 42(3), 1166-1202.

For Knockoffs Inference:

* Barber, R. F; Candès, E. J. (2015). **Controlling the false discovery rate
  via knockoffs**. *Annals of Statistics*. 43 , no. 5,
  2055--2085. doi:10.1214/15-AOS1337. https://projecteuclid.org/euclid.aos/1438606853

* Candès, E., Fan, Y., Janson, L., & Lv, J. (2018). **Panning for gold: Model-X
  knockoffs for high dimensional controlled variable selection**. *Journal of the
  Royal Statistical Society Series B*, 80(3), 551-577.

..
  Citation
  ========
  :ref:'citation'

License
=======

This project is licensed under the BSD 2-Clause License.

Acknowledgments
===============

This project has been funded by Labex DigiCosme (ANR-11-LABEX-0045-DIGICOSME)
as part of the program "Investissement d’Avenir" (ANR-11-IDEX-0003-02), by the
Fast Big project (ANR-17-CE23-0011) and the KARAIB AI Chair
(ANR-20-CHIA-0025-01). This study has also been supported by the European
Union’s Horizon 2020 research and innovation program
(Grant Agreement No. 945539, Human Brain Project SGA3).
