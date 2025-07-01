.. _quickstart_reference:

.. ## for plotting and for examples 
    #TODO Need to be updated if it's necessary
.. |MatplotlibMinVersion| replace:: 3.9.0
.. |SeabornMinVersion| replace:: 0.9.0

Quick start for contributing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. Adding it when there will a Code of Conduct
   Please note that it's very important to us that we maintain a positive and supportive
   environment for everyone who wants to participate. When you join us we ask that you follow
   our [code of conduct](https://github.com/mind-inria/hidimstat/CODE_OF_CONDUCT.rst)
   in all interactions both on and offline.

The contribution of this project will be done using Git and GitHub platform. 
If you are new to GitHub and want to learn how to fork this repo, make your own 
additions, and include those additions in the master version of this project, 
check out this `great tutorial <https://hackerbits.com/how-to-contribute-to-an-open-source-project-on-github-davide-coppola/>`__.
If you want to know about git, you can have look at this `tutorial <https://git-scm.com/docs/gittutorial>`__.


Source code
"""""""""""

You can check out the latest sources with the command:

.. code::sh
   git clone https://github.com/mind-inria/hidimstat.git

Test suite
""""""""""

For testing, we recommend you install the test dependencies with pip:

.. code::sh
   pip install hidimstat[test]
  
This will install ``pytest`` and the following extensions: 
``pytest-cov``, ``pytest-randomly``, ``pytest-xdist``, ``pytest-html``,
``pytest-timeout``, ``pytest-durations``

After this installation, you can launch the test suite::
   pytest hidimstat

Examples
""""""""

To run the examples, we recommend you install the example dependencies with pip:

.. code::sh
    pip install hidimstat[example]

For running the examples, it's necessary to install Matplotlib >= |MatplotlibMinVersion| and seaborn >=
|SeabornMinVersion|.

| After this installation, you can run any example in the `examples <https://github.com/mind-inria/hidimstat/tree/main/examples>`_ folder.
| Or you can download some of them from the `documentation <https://hidimstat.github.io/dev/auto_examples/index.html>`_.

Documentation
"""""""""""""

The documentation is built with Sphinx. We recommend you install the documentation dependencies with pip:

.. code::sh
    pip install hidimstat[doc]

After this installation, you can build the documentation from the source using the Makefile in doc_conf:

.. code::sh
    make html

``For more information``, look at the `developer documentation <https://hidimstat.github.io/dev/dev/index.html>`_