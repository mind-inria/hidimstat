.. _how_to_contribute_hidimstat:

..
  Inspired by:  
    https://skrub-data.org/stable/CONTRIBUTING.html
    https://nilearn.github.io/stable/development.html

How to contribute to HiDimStat?
###############################
First off, thanks for taking the time to contribute!

Below are some guidelines to help you get started.

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

The coding syle is ch

Coding Style
------------

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

Documentaiton style
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