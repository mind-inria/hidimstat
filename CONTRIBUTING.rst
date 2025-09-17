.. ## for plotting and for examples 
    #TODO Need to be updated if it's necessary
.. |MatplotlibMinVersion| replace:: 3.9.0
.. |SeabornMinVersion| replace:: 0.9.0

Quick start for contributing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Source code
"""""""""""

You can check out the latest sources with the command:

.. code-block:: sh

   git clone https://github.com/mind-inria/hidimstat.git

Test suite
""""""""""

For testing, we recommend you install the test dependencies with pip:

.. code-block:: sh

   pip install hidimstat[test]
  
This will install ``pytest`` and the following extensions: 
``pytest-cov``, ``pytest-randomly``, ``pytest-xdist``, ``pytest-html``,
``pytest-timeout``, ``pytest-durations``

After this installation, you can launch the test suite:

.. code-block:: sh

   pytest test

Style guide
"""""""""""

The code is formatted `black <https://black.readthedocs.io/en/stable/>`_
and `isort <https://pycqa.github.io/isort/>`_.

To apply the required dependencies and style your code:

.. code-block:: sh

    pip install -e '.[style]'
    isort .
    codespell --toml pyproject.toml
    black .

Examples
""""""""

To run the examples, we recommend you install the example dependencies with pip:

.. code-block:: sh

    pip install hidimstat[example]

For running the examples, it's necessary to install Matplotlib >= |MatplotlibMinVersion| and seaborn >=
|SeabornMinVersion|.

| After this installation, you can run any example in the `examples <https://github.com/mind-inria/hidimstat/tree/main/examples>`_ folder.
| Or you can download some of them from the `documentation <https://hidimstat.github.io/dev/generated/gallery/examples/index.html>`_.

Documentation
"""""""""""""

The documentation is built with Sphinx. We recommend you install the documentation dependencies with pip:

.. code-block:: sh

    pip install hidimstat[doc]

After this installation, you can build the documentation from the source using the Makefile in doc_conf:

.. code-block:: sh

    make html

**For more information**, look at the `developer documentation <https://hidimstat.github.io/dev/dev/index.html>`_
