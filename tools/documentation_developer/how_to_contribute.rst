.. _how_to_contribute_hidimstat:

..
  Inspired by:  
    https://skrub-data.org/stable/CONTRIBUTING.html
    https://nilearn.github.io/stable/development.html
    https://scikit-learn.org/stable/developers/contributing.html


How to contribute to HiDimStat?
###############################
First off, thanks for taking the time to contribute!

There are many ways to contribute to HiDimStat, with the most common ones
being contribution of code or documentation to the project. Improving the
documentation is no less important than improving the library itself.  If you
find a typo in the documentation, or have made improvements, do not hesitate to
create a GitHub issue or preferably submit a GitHub pull request.
Full documentation can be found under the docs/ directory.

But there are many other ways to help. In particular helping to
improve, triage, and investigate issues` and
:ref:`reviewing other developers' pull requests <review_pull_request>` are very
valuable contributions that decrease the burden on the project
maintainers.

Another way to contribute is to report issues you're facing, and give a "thumbs
up" on issues that others reported and that are relevant to you.  It also helps
us if you spread the word: reference the project from your blog and articles,
link to it from your website, or simply star to say "I use it":

.. raw:: html

  <p>
    <object
      data="https://img.shields.io/github/stars/mind-inria/hidimstat?style=for-the-badge&logo=github"
      type="image/svg+xml">
    </object>
  </p>

In case a contribution/issue involves changes to the API principles
or changes to dependencies or supported versions, it must be backed by an issue 
on the topic, see the section :ref:`Suggesting enhancement<suggesting_enhancements>`.

Have a question?
****************

You should create a `GitHub issue <https://github.com/mind-inria/hidimstat/issues>`_.

.. ADD THIS SECTION ONCE THERE WILL BE A VISION
   OR A PAPER ASSOCITE TO IT
   What to know before you begin
   *****************************
   
   To understand the purpose and goals behind HiDimStat, please read our
   :ref:`vision statement <vision>`. 

    If you're interested in the research behind HiDimStat,
    we encourage you to explore these papers:
    ADD REFERENCES

Reporting bugs
**************

Using the library is the best way to discover bugs and limitations. If you find one,
please:

1. **Check if an issue already exists**
   by searching the `GitHub issues <https://github.com/mind-inria/hidimstat/issues>`_

   - If **open**, leave a 👍 on the original message to signal that others are affected.
   - If closed, check for one of the following:
      - A **merged pull request** may indicate the bug is fixed. Update your
        HiDimStat version or note if the fix is pending a release.
      - A **wontfix label** or reasoning may be provided if the issue was
        closed without a fix.
2. If the issue does not exist, `create a new one <https://github.com/mind-inria/hidimstat/issues/new>`_.

How to submit a bug report?
^^^^^^^^^^^^^^^^^^^^^^^^^^^

To help us resolve the issue quickly, please include:

- A **clear and descriptive title**.
- A **summary of the expected result**.
- Any **additional details** where the bug might occur or doesn't occur unexpectedly.
- A **code snippet** that reproduces the issue, if applicable.
- **Version information** for Python, HiDimStat, and relevant dependencies (e.g., scikit-learn, numpy, pandas).

.. _suggesting_enhancements:

Suggesting enhancements
^^^^^^^^^^^^^^^^^^^^^^^

If you have an idea for improving HiDimStat, whether it's a small fix
or a new feature, first:

- **Check if it has been proposed or implemented** by reviewing
  `open pull requests <https://github.com/mind-inria/hidimstat/pulls?q=is%3Apr>`_.
- If not, `submit a new issue <https://github.com/mind-inria/hidimstat/issues/new>`_
  with your proposal before writing any code.

How to submit an enhancement proposal?
--------------------------------------

When proposing an enhancement:

- **Use a clear and descriptive title**.
- **Explain the goal** of the enhancement.
- Provide a **detailed step-by-step description** of the proposed change.
- **Link to any relevant resources** that may support the enhancement.


If the enhancement proposal is validated
''''''''''''''''''''''''''''''''''''''''

Once your enhancement proposal is approved, let the maintainers know the following:

- **If you will write the code and submit a Pull Request (PR)**:
  Contributing the feature yourself is the quickest way to see it implemented.
  We're here to guide you through the process if needed! To get started,
  refer to the section :ref:`writing-your-first-pull-request`.
- **If you won't be writing the code**:
  A developer can then take over the implementation.
  However, please note that we cannot guarantee how long
  it will take for the feature to be added.


If the enhancement is refused
'''''''''''''''''''''''''''''

Although many ideas are great, not all will align with the objectives
of HiDimStat.

If your enhancement is not accepted, consider implementing it
as a separate package that builds on top of HiDimStat!

We would love to see your work, and in some cases, we might even
feature your package in the official repository.

.. _review_pull_request:

Review Pull Requests
********************

Any addition to the HiDimStat's code base has to be reviewed and approved
by several people including at least one :ref:`core_devs<contact_us>`.
This can put a heavy burden on :ref:`core_devs<contact_us>` when a lot of
`pull requests <mind-inria/hidimstat/pulls>`__ are opened at the same time.
We welcome help in reviewing `pull requests <mind-inria/hidimstat/pulls>`__ from any
community member.
We do not expect community members to be experts in all changes included in `pull requests <mind-inria/hidimstat/pulls>`__,
and we encourage you to concentrate on those code changes that you feel comfortable with.
As always, more eyes on a code change means that the code is more likely to work in a wide variety of contexts!

.. _code_review:

Code Review Guidelines
^^^^^^^^^^^^^^^^^^^^^^

Reviewing code contributed to the project as PRs is a crucial component of
hidimstat development. We encourage anyone to start reviewing code of other
developers. The code review process is often highly educational for everybody
involved. This is particularly appropriate if it is a feature you would like to
use, and so can respond critically about whether the PR meets your needs. While
each pull request needs to be signed off by two core developers, you can speed
up this process by providing your feedback.

.. note::

  The difference between an objective improvement and a subjective nit isn't
  always clear. Reviewers should recall that code review is primarily about
  reducing risk in the project. When reviewing code, one should aim at
  preventing situations which may require a bug fix, a deprecation, or a
  retraction. Regarding docs: typos, grammar issues and disambiguations are
  better addressed immediately.

Important aspects to be covered in any code review
--------------------------------------------------

  Here are a few important aspects that need to be covered in any code review,
  from high-level questions to a more detailed check-list.

  - Do we want this in the library? Is it likely to be used? Do you, as
    a HiDimStat user, like the change and intend to use it? Is it in
    the scope of HiDimStat? Will the cost of maintaining a new
    feature be worth its benefits?

  - Is the code consistent with the API of HiDimStat? Are public
    functions/classes/parameters well named and intuitively designed?

  - Are all public functions/classes and their parameters, return types, and
    stored attributes named according to HiDimStat conventions and documented clearly?

  - Is any new functionality described in the user-guide and illustrated with examples?

  - Is every public function/class tested? Are a reasonable set of
    parameters, their values, value types, and combinations tested? Do
    the tests validate that the code is correct, i.e. doing what the
    documentation says it does? If the change is a bug-fix, is a
    non-regression test included? Look at `this
    <https://jeffknupp.com/blog/2013/12/09/improve-your-python-understanding-unit-testing>`__
    to get started with testing in Python.

  - Do the tests pass in the continuous integration build? If
    appropriate, help the contributor understand why tests failed.

  - Do the tests cover every line of code (see the coverage report in the build
    log)? If not, are the lines missing coverage good exceptions?

  - Is the code easy to read and low on redundancy? Should variable names be
    improved for clarity or consistency? Should comments be added? Should comments
    be removed as unhelpful or extraneous?

  - Could the code easily be rewritten to run much more efficiently for
    relevant settings?

  - Is the code backwards compatible with previous versions? (or is a
    deprecation cycle necessary?)

  - Will the new code add any dependencies on other libraries? (this is
    unlikely to be accepted)

  - Does the documentation render properly (see the
    :ref:`Documentation<contribution_documentation>` section for more details), and are the plots
    instructive?

Communication Guidelines
------------------------

  Reviewing open pull requests (PRs) helps move the project forward. It is a
  great way to get familiar with the codebase and should motivate the
  contributor to keep involved in the project. [1]_

  - Every PR, good or bad, is an act of generosity. Opening with a positive
    comment will help the author feel rewarded, and your subsequent remarks may
    be heard more clearly. You may feel good also.
  - Begin if possible with the large issues, so the author knows they've been
    understood. Resist the temptation to immediately go line by line, or to open
    with small pervasive issues.
  - Do not let perfect be the enemy of the good. If you find yourself making
    many small suggestions that don't fall into the :ref:`code_review`, consider
    the following approaches:

    - refrain from submitting these;
    - prefix them as "Nit" so that the contributor knows it's OK not to address;
    - follow up in a subsequent PR, out of courtesy, you may want to let the
      original contributor know.

  - Do not rush, take the time to make your comments clear and justify your
    suggestions.
  - You are the face of the project. Bad days occur to everyone, in that
    occasion you deserve a break: try to take your time and stay offline.

  .. [1] Adapted from the numpy `communication guidelines
        <https://numpy.org/devdocs/dev/reviewer_guidelines.html#communication-guidelines>`_.

.. _writing-your-first-pull-request:

Writing your first Pull Request
*******************************

A short summary can be find :ref:`here <quickstart_reference>`. 

Preparing the ground
^^^^^^^^^^^^^^^^^^^^^

Before writing any code, ensure you have created an issue
discussing the proposed changes with the maintainers.
See the relevant sections above on how to do this.

Setting up the environment
^^^^^^^^^^^^^^^^^^^^^^^^^^

To setup your development environment, you need to follow the steps in "From Source" tab
present in :ref:`Installing from source<installing_from_source>` page.
After that, you can return to this page to continue.

Now that the development environment is ready, you may create a new branch and start working on
the new issue.

.. code:: sh

   # fetch latest updates and start from the current head
   git fetch upstream
   git checkout -b my-branch-name-eg-fix-issue-123
   # make some changes
   git add ./the/file-i-changed
   git commit -m "my message"
   git push --set-upstream origin my-branch-name-eg-fix-issue-123

At this point, if you visit again the `pull requests
page <https://github.com/mind-inria/hidimstat/pulls>`_ github should show a
banner asking if you want to open a pull request from your new branch.


.. _implementation guidelines:

Implementation Guidelines
^^^^^^^^^^^^^^^^^^^^^^^^^

When contributing, keep these project goals in mind:

- **Pure Python code**: Don't use any binary extensions, Cython, or other compiled languages.
- **Production-friendly code**:
    - Target the widest possible range of Python versions and dependencies.
    - Minimize the use of external dependencies.
    - Ensure backward compatibility as much as possible.
- **Performance over readability**:
  Optimized code may be less readable, so please include clear and detailed comments.
  Refer to this `best practice guide <https://stackoverflow.blog/2021/12/23/best-practices-for-writing-code-comments/>`_.
- **Explicit variable/function names**: Use descriptive, verbose names for clarity.
- **Document public API components**:
    - Document all public functions, methods, variables, and class signatures.
    - The public API refers to all components available for import and use by library users. Anything that doesn't begin with an underscore is considered part of the public API.

Coding Style
^^^^^^^^^^^^

The HiDimStat codebase follows `PEP8 <https://peps.python.org/pep-0008/>`__ styling.


The main conventions we enforce are :

- line length < 80
- spaces around operators
- meaningful variable names
- function names are underscore separated (e.g., ``a_nice_function``) and as short as possible
- public functions exposed in their parent module's init file
- private function names preceded with a "_" and very explicit
- classes in CamelCase
- 2 empty lines between functions or classes
- 4 spaces instead of tabs per indentation level
- 

You can check that any code you may have edited follows these conventions
by running `black <https://black.readthedocs.io/en/stable/>`__.


Testing the code
^^^^^^^^^^^^^^^^

Tests for files in a given folder should be located in the folder named ``tests`` 
with a tree structure that reflects the source folder. For example, the tests 
for the functions in `src/hidimstat/statistical_tools/p_values.py` are in 
`test\statistical_tools\test_p_values.py`. 

Tests should check all functionalities of the code that you are going to
add. If needed, additional tests should be added to verify that other
objects behave correctly.

.. If there is any dependance between the tests or function, we need to write
   an example of it here. 

Run each updated test file using ``pytest``
(`pytest docs <https://docs.pytest.org/en/stable>`_):

.. code:: sh

   pytest -vsl tests/test_amazing_method.py

The ``-vsl`` flag provides more information when running the tests.

It is also possible to run a specific test, or set of tests using the
commands ``pytest the_file.py::the_test``, or
``pytest the_file.py -k 'test_name_pattern'``. This is helpful to avoid
having to run all the tests.

If you work on Windows, you might have some issues with the working
directory if you use ``pytest``, while ``python -m pytest ...`` should
be more robust.

Once you are satisfied with your changes, you can run all the tests to make sure
that your change did not break code elsewhere:

.. code:: sh

    pytest -s tests

Finally, sync your changes with the remote repository and wait for CI to run.

Checking coverage on the local machine
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Checking coverage is one of the operations that is performed after
submitting the code. As this operation may take a long time online, it
is possible to check whether the code coverage is high enough on your
local machine.

Run your tests with the ``--cov`` and ``--cov-report`` arguments:

.. code:: sh

   pytest -vsl tests/test_amazing_method.py --cov=. --cov-report=html

This will create the folder ``htmlcov``: by opening
``htmlcov/index.html`` it is possible to check what lines are covered in
each file.

.. ADD THIS SECTION WHEN THERE WILL BE DOCTRING TESTS
   Updating doctests
   ^^^^^^^^^^^^^^^^^

   If you alter the default behavior of an object, then this might affect
   the docstrings. Check for possible problems by running

   .. code:: sh

      pytest src/hidimstat/path/to/file


.. ADD THIS SECTION WHEN THERE WILL BE PRECOMMIT CONFIGURATION FILE
   Formatting and pre-commit checks
   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

   Formatting the code well helps with code development and maintenance,
   which why is HiDimStat requires that the last commits of a pull request follow
   a specific set of formatting rules to ensure code quality.

   Luckily, these checks are performed automatically by the ``pre-commit``
   tool (`pre-commit docs <https://pre-commit.com>`_) before any commit
   can be pushed. Something worth noting is that if the ``pre-commit``
   hooks format some files, the commit will be canceled: you will have to
   stage the changes made by ``pre-commit`` and commit again.

Submitting your code
--------------------

Once you have pushed your commits to your remote repository, you can submit
a PR by clicking the "Compare & pull request" button on GitHub,
targeting the HiDimStat repository.

.. _commit_markers:

Commit message markers
^^^^^^^^^^^^^^^^^^^^^^

Please note that if one of the following markers appears in the latest commit
message, the following actions are taken.

.. This should be in coherence with the CI_documentation

====================== ===================
Commit Message Marker  Action Taken by CI
---------------------- -------------------
[skip tests]           The tests are not run
[doc skip]             Docs are not built
[doc quick]            Docs built, but excludes example gallery plots
[doc change]           Docs built, including only example, which has been directly modified
====================== ===================

Continuous Integration (CI)
^^^^^^^^^^^^^^^^^^^^^^^^^^^
After creating your PR, CI tools will run proceed to run all the tests on all
configurations supported by HiDimStat.

- **Github Actions**:
  Used for testing HiDimStat across various platforms (Linux, macOS, Windows)
  and dependencies.
- **CircleCI**:
  Builds and verifies the project documentation.

For more details, look at the :ref:`CI description <developer_documentation_CI>`


Integration
-----------

Community consensus is key in the integration process. Expect a minimum
of 1 to 3 reviews depending on the size of the change before we consider
merging the PR.

.. _quick_start_build_doc:

Building the documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^
..
  Inspired by: https://github.com/scikit-learn/scikit-learn/blob/main/doc/developers/contributing.rst

**Before submitting your pull request, ensure that your modifications haven't
introduced any new Sphinx warnings by building the documentation locally
and addressing any issues.**

First, make sure you have properly installed the development version of HiDimStat.
You can follow the :ref:`installation <installing_from_source>` from source section, if needed.

To build the documentation, you need to be in the ``docs`` folder:

.. code:: bash

    cd docs

To generate the full documentation, including the example gallery,
run the following command:

.. code:: bash

    make html

For more information, look at the page `Building the documentation<developer_documentation_build>`

.. _contribution_documentation:

Documentation
*************

Editing the API reference documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To add a new entry to the :ref:`API reference documentation<api_documentation>` or change its
content, head to ``docs/src/api_reference.py``. This data is then used by ``docs/tools/conf.py``
to render templates located at ``doc/tools/_templates/*.rst``.

Note that **all public functions and classes must be documented in the API
reference**, hence when adding a public function or class, a new entry must be
added, as detailed just above.

Docstring style
^^^^^^^^^^^^^^^

Each function and class must come with a “docstring” at the top of the function code,
using a forating close to the `numpydoc <https://numpydoc.readthedocs.io/en/latest/format.html>`__ formatting.
The docstring must summarize what the function does and document every parameter.

If an argument takes in a default value, it should be described
with the type definition of that argument.

See the examples below:

.. code-block:: python

      def good(x, y=1, z=None):
          """Show how parameters are documented.

          Parameters
          ----------
          x : int
                X

          y : int, default=1
                Note that "default=1" is preferred to "Defaults to 1".

          z : str, default=None

          """


      def bad(x, y=1, z=None):
          """Show how parameters should not be documented.

          Parameters
          ----------
          x :
              The type of X is not described

          y : int
              The default value of y is not described.

          z : str
              Defaults=None.
              The default value should be described after the type.
          """

Additionally, we consider it best practice to write modular functions;
i.e., functions should preferably be relatively short and do *one* thing.
This is also useful for writing unit tests.

Writing small functions is not always possible, and we do not recommend 
trying to reorganize larger, but well-tested, older functions in the codebase, 
unless there is a strong reason to do so (e.g., when adding a new feature).

Documentation style
^^^^^^^^^^^^^^^^^^^

Documentation must be understandable by people from different backgrounds.
The “narrative” documentation should be an introduction to the concepts of
the library.
It includes very little code and should first help the user figure out which
parts of the library he needs and then how to use it.
It must be full of links, of easily-understandable titles, colorful boxes and
figures.

Examples take a hands-on approach focused on a generic usecase from which users
will be able to adapt code to solve their own problems.
They include plain text for explanations, python code and its output and
most importantly figures to depict its results.
Each example should take only one or two minutes to run.

To build our documentation, we are using
`sphinx <https://www.sphinx-doc.org/en/master/usage/quickstart.html>`__ for the
main documentation and
`sphinx-gallery <https://sphinx-gallery.github.io/stable/index.html>`__ for the
example tutorials. If you want to work on those, check out next section to
learn how to use those tools to :ref:`build documentation<quick_start_build_doc>`.


Bibliography reference
^^^^^^^^^^^^^^^^^^^^^^

All the bibliography references are included in the file references.bib in tools.
This file contains the reference under the format of `bibtext <https://www.bibtex.com/g/bibtex-format/>`__.
There are ordered by alphabetic order of the name of the first authors and the years.
It's recommended to provide the reference to the paper and not the associate 
preprint for all implemented methods.
