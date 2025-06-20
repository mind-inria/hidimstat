.. ## Define a hard line break in HTML
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

=================================================================
HiDimStat: High-dimensional statistical inference tool for Python
=================================================================

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

Visit our website, https://hidimstat.github.io/, for more information.

..
  ## TODO: Add short citation when this will be ready
  If you use HiDimStat for your published research, we kindly ask you to :ref:`cite<citation>` our article:
  short reference

Find your important variables in your data with the help of 
`our examples <https://hidimstat.github.io/dev/auto_examples/index.html>`_.

If you have any problems, please report them to the `GitHub issue tracker <https://github.com/mind-inria/hidimstat/issues>`_ 
or contribute to the library by opening a pull request.

Installation
------------

Dependencies
^^^^^^^^^^^^

.. # Add dependency of the project
  TODO Need to match with pyproject.toml

HiDimStat requires:

- Python (>= |PythonMinVersion|)
- joblib (>= |JoblibMinVersion|)
- NumPy (>= |NumPyMinVersion|)
- Pandas (>= |PandasMinVersion|)
- Scikit-learn (>= |SklearnMinVersion|)
- SciPy (>= |SciPyMinVersion|)

HiDimStat's plotting capabilities require Matplotlib (>= |MatplotlibMinVersion|).

To run the examples, Matplotlib (>= |MatplotlibMinVersion|) and seaborn (>=
|SeabornMinVersion|) are required.

User installation
^^^^^^^^^^^^^^^^^

.. # Add the instruction for installation
  TODO add conda when it will be accessible

HiDimStat can easily be installed via ``pip``. For more installation information,
see the `installation instructions <https://hidimstat.github.io/dev/index.html#installation>`_::

   pip install -U hidimstat 

Contribute
----------

.. # Add short discription for contribution to the library

The best way to support the development of HiDimStat is to spread the word!

HiDimStat aims to be supported by an active community, and we welcome 
contributions to our code and documentation.

For bug reports, feature requests, documentation improvements, or other issues, 
you can create a `GitHub issue <https://github.com/mind-inria/hidimstat/issues>`_.

If you want to contribute directly to the library, check the 
`how to contribute <https://hidimstat.github.io/dev/How-to-Contribute/>`_ page 
on the website for more information.

Quick start for development
^^^^^^^^^^^^^^^^^^^^^^^^^^^
Source code
"""""""""""

You can check out the latest sources with the command::

   git clone https://github.com/mind-inria/hidimstat.git

Test suite
""""""""""

For testing, we recommend you install the test dependencies with pip::
   pip install hidimstat[test]
  
This will install ``pytest`` and the following extensions: 
``pytest-cov``, ``pytest-randomly``, ``pytest-xdist``, ``pytest-html``,
``pytest-timeout``, ``pytest-durations``

After this installation, you can launch the test suite::
   pytest hidimstat

Examples
""""""""

To run the examples, we recommend you install the example dependencies with pip::
    pip install hidimstat[example]

For running the examples, it's necessary to install Matplotlib >= |MatplotlibMinVersion| and seaborn >=
|SeabornMinVersion|.

| After this installation, you can run any example in the `examples <https://github.com/mind-inria/hidimstat/tree/main/examples>`_ folder.
| Or you can download some of them from the `documentation <https://hidimstat.github.io/dev/>`_.

Documentation
"""""""""""""

The documentation is built with Sphinx. We recommend you install the 
 documentation dependencies with pip::
    pip install hidimstat[doc]

After this installation, you can build the documentation from the source using 
 the Makefile in doc_conf::
    make html

Contact us
----------

.. # Add a way to contact maintainers 
  TODO this needs to be updated when there is a change of maintainers

Currently, this library is supported by the `INRIA <https://www.inria.fr/en>`_ 
team `MIND <https://www.inria.fr/fr/mind>`_. |br|
If you want to report a problem or suggest an enhancement, we would love for you 
to `open an issue <https://github.com/mind-inria/hidimstat/issues/new>`_ at 
this GitHub repository so we can address it quickly. |br|
For less formal discussions or to exchange ideas, you can contact the main 
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

Citation
--------

If you use a HiDimStat method for your research, you'll find the associated 
reference paper in the method description, and we recommend that you cite it.

..
  TODO add the section for citing the library once a Zenodo repository is made
  or a paper is published.

If you publish a paper using HiDimStat, please contact us or open an issue! 
We would love to hear about your work and help you promote it.

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
(ANR-20-CHIA-0025-01), and by the `VITE project <https://anr.fr/Projet-ANR-23-CE23-0016>`_ (ANR-23-CE23-0016).
This study has also been supported by the European Union’s Horizon 2020 research and innovation program 
as part of the program `Human Brain Project SGA3 <https://cordis.europa.eu/project/id/945539>`_
(Grant Agreement No. 945539) and `EBRAIN-Health <https://cordis.europa.eu/project/id/101058516>`_ 
(Grant Agreement No. 101058516).
