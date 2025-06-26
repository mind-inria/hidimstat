
Quick start for contributing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Source code
"""""""""""

You can check out the latest sources with the command:
::
   git clone https://github.com/mind-inria/hidimstat.git

Test suite
""""""""""

For testing, we recommend you install the test dependencies with pip:
::
   pip install hidimstat[test]
  
This will install ``pytest`` and the following extensions: 
``pytest-cov``, ``pytest-randomly``, ``pytest-xdist``, ``pytest-html``,
``pytest-timeout``, ``pytest-durations``

After this installation, you can launch the test suite:
::
   pytest hidimstat

Examples
""""""""

To run the examples, we recommend you install the example dependencies with pip:
::
    pip install hidimstat[example]

For running the examples, it's necessary to install Matplotlib >= |MatplotlibMinVersion| and seaborn >=
|SeabornMinVersion|.

| After this installation, you can run any example in the `examples <https://github.com/mind-inria/hidimstat/tree/main/examples>`_ folder.
| Or you can download some of them from the `documentation <https://hidimstat.github.io/dev/auto_examples/index.html>`_.

Documentation
"""""""""""""

The documentation is built with Sphinx. We recommend you install the 
documentation dependencies with pip:
::
    pip install hidimstat[doc]

After this installation, you can build the documentation from the source using 
the Makefile in doc_conf:
::
    make html