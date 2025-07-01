.. _installation_instructions:
..
  Inspired by:  https://skrub-data.org/stable/install.html

.. currentmodule:: hidimstat

=======
Install
=======

.. raw:: html

    <div class="container mt-4">

    <ul class="nav nav-pills nav-fill" id="installation" role="tablist">
        <li class="nav-item" role="presentation">
            <a class="nav-link active" id="pip-tab" data-bs-toggle="tab" data-bs-target="#pip-tab-pane" type="button" role="tab" aria-controls="pip" aria-selected="true">Using pip</a>
        </li>
        <--- <li class="nav-item" role="presentation">
            <a class="nav-link" id="conda-tab" data-bs-toggle="tab" data-bs-target="#conda-tab-pane" type="button" role="tab" aria-controls="conda" aria-selected="false">Using conda</a>
        </li> --->
        <li class="nav-item" role="presentation">
            <a class="nav-link" id="source-tab" data-bs-toggle="tab" data-bs-target="#source-tab-pane" type="button" role="tab" aria-controls="source" aria-selected="false">From source</a>
        </li>
    </ul>

    <div class="tab-content">
        <div class="tab-pane fade show active" id="pip-tab-pane" role="tabpanel" aria-labelledby="pip-tab" tabindex="0">
            <hr />

.. code:: console

    pip install hidimstat -U

.. ADD WHEN ADDED CONDA
   .. raw:: html
    </div>
        <div class="tab-pane fade" id="conda-tab-pane" role="tabpanel" aria-labelledby="conda-tab" tabindex="0">
        <hr />
    .. code:: console

        conda install -c conda-forge hidimstat

.. raw:: html

        </div>
        <div class="tab-pane fade" id="source-tab-pane" role="tabpanel" aria-labelledby="source-tab" tabindex="0">
            <hr />

.. _installing_from_source:

Advanced Usage for Contributors
-------------------------------

1. Fork the project
'''''''''''''''''''

To contribute to the project, you first need to
`fork HiDimStat on GitHub <https://github.com/mind-inria/hidimstat/fork>`_.

That will enable you to push your commits to a branch *on your fork*.

2. Clone your fork
''''''''''''''''''

Clone your forked repo to your local machine:

.. code:: console

    git clone https://github.com/<YOUR_USERNAME>/hidimstat
    cd hidimstat

Next, add the *upstream* remote (i.e. the official HiDimStat repository). This allows you
to pull the latest changes from the main repository:

.. code:: console

    git remote add upstream https://github.com/mind-inria/hidimstat.git

Verify that both the origin (your fork) and upstream (official repo)
are correctly set up:

.. code:: console

    git remote -v

You should see something like this:

.. code:: console

    origin  git@github.com:<YOUR_USERNAME>/hidimstat.git (fetch)
    origin  git@github.com:<YOUR_USERNAME>/hidimstat.git (push)
    upstream        git@github.com:mind-inria/hidimstat.git (fetch)
    upstream        git@github.com:mind-inria/hidimstat.git (push)


3. Setup your environment
'''''''''''''''''''''''''

Now, setup a development environment.
You can set up a virtual environment with Conda, or with python's ``venv``:

.. ADD WHEN CONDA WILL POSSIBLE
   - With `conda <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`__:
   .. code:: console
       conda create -n env_hidimstat python=3.13
       conda activate env_hidimstat

- With `venv <https://docs.python.org/3/library/venv.html>`__:

.. code:: console

    python -m venv env_hidimstat
    source hidimstat/bin/activate

Then, with the environment activated and at the root of your local copy of HiDimStat,
install the local package in editable mode with development dependencies:

.. code:: console

    pip install -e ".[dev]"

.. ADD WHEN PROCOMMIT CONFIGURATION FILE WILL BE READY
   Enabling pre-commit hooks ensures code style consistency by triggering checks (mainly formatting) every time you run a ``git commit``.
    .. code:: console
        pre-commit install


Optionally, configure Git to ignore certain revisions in git blame and
IDE integrations. These revisions are listed in .git-blame-ignore-revs:

.. code:: console

    git config blame.ignoreRevsFile .git-blame-ignore-revs

4. Run the tests
''''''''''''''''

To ensure your environment is correctly set up, run the test suite:

.. code:: console

    pytest --pyargs hidimstat

Testing should take less than 5 minutes.

After that, your environment is ready for development!

Now that you're set up,
you may return to :ref:`writing your first pull request<writing-your-first-pull-request>`
and start coding!

.. raw:: html

        </div>
    </div>
    </div>
