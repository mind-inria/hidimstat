.. _dev_contributing:
.. ## for plotting and for examples
    #TODO Need to be updated if it's necessary
.. |MatplotlibMinVersion| replace:: 3.9.0
.. |SeabornMinVersion| replace:: 0.9.0

How to contribute to Hidimstat
==============================

.. contents:: On this page
   :local:
   :depth: 2

Quick start for contributing
----------------------------
Source code
~~~~~~~~~~~

You can check out the latest sources with the command:

.. code-block:: sh

   git clone https://github.com/mind-inria/hidimstat.git

Test suite
~~~~~~~~~~

For testing, we recommend you install the test dependencies with pip:

.. code-block:: sh

   pip install -e '.[test]'

This will install ``pytest`` and the following extensions:
``pytest-cov``, ``pytest-randomly``, ``pytest-xdist``, ``pytest-html``,
``pytest-timeout``, ``pytest-durations``

After this installation, you can launch the test suite:

.. code-block:: sh

   pytest test

Style guide
~~~~~~~~~~~

The code is formatted and linted with with `ruff <https://github.com/astral-sh/ruff>`_.

Spelling is checked using `codespell <https://github.com/codespell-project/codespell>`_.

To apply the required dependencies and style your code:

.. code-block:: sh

    pip install -e '.[style]'
    ruff format .
    ruff check --fix .
    codespell --toml pyproject.toml

Examples
~~~~~~~~

To run the examples, we recommend you install the example dependencies with pip:

.. code-block:: sh

    pip install -e '.[example]'

For running the examples, it's necessary to install Matplotlib >= |MatplotlibMinVersion| and seaborn >=
|SeabornMinVersion|.

| After this installation, you can run any example in the `examples <https://github.com/mind-inria/hidimstat/tree/main/examples>`_ folder.
| Or you can download some of them from the `documentation <https://hidimstat.github.io/dev/generated/gallery/examples/index.html>`_.

Documentation
~~~~~~~~~~~~~

The documentation is built with Sphinx. We recommend you install the documentation dependencies with pip:

.. code-block:: sh

    pip install -e '.[doc]'

After this installation, you can build the documentation from the source using the Makefile in doc_conf:

.. code-block:: sh

    make html

**For more information**, look at the `developer documentation <https://hidimstat.github.io/dev/dev/index.html>`_

Github repository guidelines
----------------------------
Issues
~~~~~~

One way of starting to contribute to the hidimstat project is by opening up issues on the Github about
new feature ideas, documentation improvement, or bug fixes. Please check beforehand that no similar
issue exists. If so, please feel free to contribute to the existing discussion while respecting our
:ref:`code of conduct<dev_code_of_conduct>`.

When opening issues, we recommend that you include a tag at the beginning of the title
so that contributors may quickly identify the nature of the issue. Here are the currently
used tags:

* [DOC]: for documentation needs
* [FEAT]: to request a new feature
* [BUG]: to report a bug
* [ENH]: to suggest enhancements, improvements of existing code
* [TEST]: to discuss about tests
* [VIS]: to suggest improvements for visualization
* [MAINT]: for maintenance needs

Pull-requests
~~~~~~~~~~~~~

Another way of contributing is by submitting changes to the project via Github pull-requests.
Pull-requests should only be open to suggest changes according to existing issues. If you don't know
how to do so, please check the Github explanations on `pull requests linked to issues <https://docs.github.com/en/issues/tracking-your-work-with-issues/using-issues/linking-a-pull-request-to-an-issue>`_.
Please create a new branch to work on the changes before opening the pull-request. Two reviewers will
examine the submitted changes, and approve or discuss modifications. The changes can be merged to the main
branch after the approval by two reviewers, and after all tests automatically launched by the :ref:`Continuous
Integration<developer_documentation_CI>` processes have passed.

Pull-requests' titles should follow the same naming convention as issues.

To help us keep track of all changes from a version to another, we recommend that developers fill the ``CHANGELOG.rst`` file
with their changes before opening the pull-request, and fill in their information in the ``CITATION.cff`` file if it has not
been done yet.

Repository folder architecture
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here is the general organization of the repository`s folders:

    =============== ===========
    **Folder name** **What the folder contains**
    --------------- -----------
    .circleci       The configuration file for the :ref:`Continuous Integration <developer_documentation_CI>` tool CircleCI.
    .github         All configuration files for `Github Actions <https://github.com/mind-inria/hidimstat/actions>`_.
    docs            All files about the API, User guide, the examples gallery, glossary and notations.
    examples        Python files that run for the examples gallery.
    src/hidimstat   The Python files of the HiDimStat library.
    test            The Python files that run tests on the library.
    tools           The developer documentation files.
    =============== ===========

We give more details about the organization of the src/hidimstat folder:

    ================= ===========
    **Folder name**   **What the folder contains**
    ----------------- -----------
    base              Base variable importance (VI) and perturbation classes, all derived VI and perturbation methods as classes, as well as inference classes.
    samplers          Sampling classes.
    statistical_tools Statistical tools for tests, compute p-values, aggregation methods.
    visualization     Visualization methods to produce graphs.
    _utils            Private folder that contains utility functions and classes for internal usage only.
    ================= ===========

AI-assisted tools
~~~~~~~~~~~~~~~~~

While we understand the benefits of using AI-assisted tools to write code, we kindly ask you to
double check the code, and be prepared to explain it upon request during review. If the submitted code
was written with the assistance of AI, we also request that you state it in your PR, and certify that
you have read and understood it. We also kindly ask you to refrain from using AI tools to generate
comments on issues or PRs.

Code Writing Conventions
------------------------

Naming
~~~~~~
Files should follow the `snake case <https://en.wikipedia.org/wiki/Snake_case>`_ convention,
and classes should follow the `upper camel case <https://en.wikipedia.org/wiki/Camel_case>`_
convention. Names should not contain acronyms and should instead include the full version,
except for really long names, such as Distilled Conditional Randomization Test
abbreviated as D0CRT, or names that are standard in the scientific community, such as
LOCO which stands for Leave One Covariate Out. Functions should be written in snake case,
and should not contain acronyms similarly to classes.

Classes
~~~~~~~
While we do not provide any template for the classes, we try to follow the conventions on
`developing scikit-learn estimators <https://scikit-learn.org/stable/developers/develop.html>`_.
We recommend that you follow the general style of the ``BaseVariableImportance`` class when
implementing other classes. The main addition from scikit-learn's :py:class:`sklearn.base.BaseEstimator` is the
addition of importance and selection methods that allow the selection of variables based on
different criteria.


Citations
~~~~~~~~~
When introducing citations in code documentation, please use the following format:

.. code-block::

  # :footcite:t: \`nameofthecitation\`

Make sure that the citation name is included in the bibliography and that it is listed in alphabetical order.

Reproducibility and Randomness
------------------------------

Design choice: :py:class:`numpy.random.Generator`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Hidimstat uses numpy.random.Generator for all randomness because it provides a reliable
way to manage separate, reproducible streams of random numbers. The key benefit is the
ability to spawn new, independent Generator instances from a single parent.

This is particularly useful for our methods that employ parallel samplers or other
algorithms requiring multiple, distinct random streams. Spawning ensures that each
parallel process has its own unique sequence of random numbers, preventing interference
and guaranteeing that the overall result remains reproducible.

The Generator object for any given method is instantiated by the internal
``check_random_state`` utility function, which handles the logic based on the user's
input.


The random_state Parameter
~~~~~~~~~~~~~~~~~~~~~~~~~~

The behavior of a method's randomness is exclusively controlled by the random_state
parameter. It accepts three types of input:

* :py:class:`int`:
  An integer input ensures both **reproducibility**: running the script twice
  produces identical results, and **repeatability**: calling the method twice within the
  same script, such as ``<method>.importance()``, yields the exact same output.
* :py:class:`numpy.random.Generator`:
  Passing a Generator object provides reproducibility but not
  repeatability. The stream of random numbers is reproducible if the user seeds their
  Generator object, but subsequent calls to the method will advance the generator's
  state, leading to different results.
* :py:class:`None`:
  This input provides neither reproducibility nor repeatability. The random
  numbers are generated from a non-deterministic source, resulting in different outcomes
  on each run and each call.
