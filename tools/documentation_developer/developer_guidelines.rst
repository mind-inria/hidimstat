.. _developer_guidelines:

Developer Guidelines
====================

This section provides guidelines for developers contributing to the hidimstat project.

.. contents:: Table of Contents
   :depth: 2
   :local:

Repository folder architecture
------------------------------

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

Writing Conventions
-------------------

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
`developping scikit-learn estimators <https://scikit-learn.org/stable/developers/develop.html>`_.
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


Testing
~~~~~~~
All methods that involve randomness must include a dedicated test suite. The following
test patterns help to ensure consistent behavior:

* ``test_<method>_repeatability``
* ``test_<method>_randomness_with_none``
* ``test_<method>_reproducibility_with_integer``
* ``test_<method>_reproducibility_with_rng``
