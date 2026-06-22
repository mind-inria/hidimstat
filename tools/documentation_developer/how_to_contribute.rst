.. _how_to_contribute_hidimstat:

How to contribute to HiDimStat?
===============================

This section explains how developers can contribute to the project, and the general workflow of the hidimstat project.

.. contents:: Table of Contents
   :depth: 2
   :local:

Issues
------

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
-------------

Another way of contributing is by submitting changes to the project via Github pull-requests.
Pull-requests should only be open to suggest changes according to existing issues. If you don't know
how to do so, please check the Github explanations on `pull requests linked to issues <https://docs.github.com/en/issues/tracking-your-work-with-issues/using-issues/linking-a-pull-request-to-an-issue>`_.
Please create a new branch to work on the changes before opening the pull-request. Two reviewers will
examine the submitted changes, and approve or discuss modifications. The changes can be merged to the main
branch after the approval by two reviewers, and after all tests automatically launched by the :ref:`Continuous
Integration<developer_documentation_CI>` processes have passed.

Pull-requests' titles should follow the same naming convention as issues.

AI-assisted tools
-----------------

While we understand the benefits of using AI-assisted tools to write code, we kindly ask you to
double check the code, and be prepared to explain it upon request during review. If the submitted code
was written with the assistance of AI, we also request that you state it in your PR, and certify that
you have read and understood it. We also kindly ask you to refrain from using AI tools to generate
comments on issues or PRs.

Other
-----

To help us keep track of all changes from a version to another, we recommend that developers fill the ``CHANGELOG.rst`` file
with their changes before opening the pull-request, and fill in their information in the ``CITATION.cff`` file if it has not
been done yet.
