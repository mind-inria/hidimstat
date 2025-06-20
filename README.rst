
.. ## define a hard line break for HTML
.. |br| raw:: html

   <br />

.. # Add the reference for the badges
.. ## Reference to the CI status
.. |Linter&Tests| image:: https://github.com/mind-inria/hidimstat/actions/workflows/main_workflow.yml/badge.svg?branch=main
    :target: https://github.com/mind-inria/hidimstat/actions/workflows/main_workflow.yml?query=branch%3Amain
.. |CircleCI/Documentation| image:: https://circleci.com/gh/mind-inria/hidimstat.svg?style=shield
    :target: https://circleci.com/gh/mind-inria/hidimstat?branch=main
.. |CodeCov| image:: https://codecov.io/github/mind-inria/hidimstat/branch/main/graph/badge.svg?token=O1YZDTFTNS
    :target: https://codecov.io/github/mind-inria/hidimstat
.. ## Distribution python
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
.. ## Additional badge
.. |GitHub contributors| image:: https://img.shields.io/github/contributors/mind-inria/hidimstat.svg?logo=github
  :target: https://github.com/mind-inria/hidimstat
.. |GitHub commit activity| image:: https://img.shields.io/github/commit-activity/y/mind-inria/hidimstat.svg?logo=github&color=%23ff6633
  :target: https://github.com/mind-inria/hidimstat
.. |License| image:: https://img.shields.io/github/license/mind-inria/hidimstat
    :target: https://opensource.org/license/bsd-3-clause
.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black

.. # Add minimal dependecy of the main packages
   ## This need to update in same time that pyproject.toml
.. |PythonMinVersion| replace:: 3.9
.. |JoblibMinVersion| replace:: 1.2
.. |NumPyMinVersion| replace:: 1.25
.. |PandasMinVersion| replace:: 2.0
.. |SklearnMinVersion| replace:: 1.4
.. |SciPyMinVersion| replace:: 1.6
.. ## for plotting and for examples
.. |MatplotlibMinVersion| replace:: 3.9.0
.. |SeabornMinVersion| replace:: 0.9.0

*****************************************************************
HiDimStat: High-dimensional statistical inference tool for Python
*****************************************************************

.. # Add the different badge

|Linter&Tests| |CircleCI/Documentation| |CodeCov|

|PyPi| |PyPi format| |PyPi_download| |Latest release| |PythonVersion|

|GitHub contributors| |GitHub commit activity| |License| |Black|

.. # Short description of the library

The HiDimStat package provides statistical inference methods to solve the problem
of support recovery in the context of high-dimensional and spatially structured data.

.. # Add usefull links

If you like the package, spread the word and ⭐ our `official repository 
<https://github.com/mind-inria/hidimstat>`_!

Visit your website, https://hidimstat.github.io/, for more information.

..
  ## TODO: Add short citation when this will be ready
  If you use HiDimStat for your published research, we kindly ask you to :ref:`cite<citation>` our article:
  short reference

Find your important variables in your data with the help of 
`our examples <https://hidimstat.github.io/dev/auto_examples/index.html>`_.

If you have any problems, please report them to the github issue tracker 
(https://github.com/mind-inria/hidimstat/issues) or contribute to the library 
by opening a Pull Request.

Installation
------------

Dependencies
~~~~~~~~~~~~

.. # Add dependency of the project
   Need to match with pyproject.toml

HiDimStat requires:

- Python (>= |PythonMinVersion|)
- joblib (>= |JoblibMinVersion|)
- NumPy (>= |NumPyMinVersion|)
- Pandas (>= |PandasMinVersion|)
- Scikit-learn (>= |SklearnMinVersion|)
- SciPy (>= |SciPyMinVersion|)

=======

HiDimStat plotting capabilities require Matplotlib (>= |MatplotlibMinVersion|).
For running the examples Matplotlib >= |MatplotlibMinVersion| and seaborn >=
|SeabornMinVersion| is required.

=======

For testing HiDimStat, it require  ``pytest`` and the following extension: 
``pytest-cov``, ``pytest-randomly``, ``pytest-xdist``, ``pytest-html``,
``pytest-timeout``, ``pytest-durations``

User installation
~~~~~~~~~~~~~~~~~

.. # Add the instruction for installation
   TODO add conda when it will be accessible

HiDimStat can easily be installed via ``pip``. For more installation information,
see the `installation instructions <https://hidimstat.github.io/dev/index.html#installation>`_.::

    pip install -U hidimstat 

Contribute
----------

.. # Add short discription for contribution to the library

The best way to support the development of HiDimStat is to spread the word!

HiDimStat wants to be supported by an active community and we welcome 
contributions to our code and documentation.

For bug reports, feature requests, documentation improvements, or other issues, 
you can create a `GitHub issue <https://github.com/mind-inria/hidimstat/issues>`_.

If you want to contribute directly to the library, then check the 
`how to contribute <https://hidimstat.github.io/dev/How-to-Contribute/>`_ page 
son the website for more information.

Contact us
----------

.. # Add a way to contact mainteners 
   TODO this need to be updated when there will a change of mainteners

Actually, this library is supported by the `INRIA <https://www.inria.fr/en>`_ 
team `MIND <https://www.inria.fr/fr/mind>`_. |br|
If you want to report a problem or suggest an enhancement, we had love for you 
to `open an issue <https://github.com/mind-inria/hidimstat/issues/new>`_ at 
this GitHub repository because then we can get right on it. |br|
For a less formal discussion or exchanging ideas, you can contact the main 
contributors:

+------------------+------------------+------------------+
|   Lionel Kusch   | Bertrand Thirion |  Joseph Paillard |
+------------------+------------------+------------------+
|    |avatar LK|   |   |avatar BT|    |    |avatar JP|   |
+------------------+------------------+------------------+

.. |avatar LK| image:: https://avatars.githubusercontent.com/u/17182418?v=4 
  :target: https://github.com/lionelkusch
.. |avatar BT| image:: https://avatars.githubusercontent.com/u/234454?v=4
  :target: https://github.com/bthirion
.. |avatar JP| image:: https://avatars.githubusercontent.com/u/56166877?v=4 
  :target: https://github.com/jpaillard

..
  Citation
  ========
  :ref:'citation'
  TODO add the section for citing the library once a zenodo repository is made
  or a paper is published.

License
-------

This project is licensed under the 
`BSD 3-Clause License <https://github.com/mind-inria/hidimstat?tab=BSD-3-Clause-1-ov-file>`_.

Acknowledgments
---------------

This project has been funded by `Labex DigiCosme <https://anr.fr/ProjetIA-11-LABX-0045>`_
(ANR-11-LABEX-0045-DIGICOSME) as part of the program 
`Investissement d’Avenir <https://anr.fr/ProjetIA-11-IDEX-0003>`_ 
(ANR-11-IDEX-0003-02), by the `Fast Big project <https://anr.fr/Projet-ANR-17-CE23-0011>`_
(ANR-17-CE23-0011), by the `KARAIB AI Chair <https://anr.fr/Projet-ANR-20-CHIA-0025>`_ 
(ANR-20-CHIA-0025-01) and by the `VITE project <https://anr.fr/Projet-ANR-23-CE23-0016>`_ (ANR-23-CE23-0016).
This study has also been supported by the European Union’s Horizon 2020 research and innovation program 
as part of the program `Human Brain Project SGA3 <https://cordis.europa.eu/project/id/945539>`_
(Grant Agreement No. 945539) and `EBRAIN-Health <https://cordis.europa.eu/project/id/101058516>`_ 
(Grant Agreement No. 101058516).

