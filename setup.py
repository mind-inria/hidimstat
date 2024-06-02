#!/usr/bin/env python

import os
import sys

from setuptools import find_packages

PKG = 'hidimstat'
DESCRIPTION = "High-dimensional statistical inference tools for Python"
LONG_DESCRIPTION = open('README.md').read()
MAINTAINER = 'Chevalier (ja-che), Nguyen (tbng), Alexandre Blain (alexblnn), Ahmad Chamma (achamma723) and Bertrand Thirion (bthirion)'
MAINTAINER_EMAIL = 'bertrand.thirion@inria.fr'
URL = 'https://github.com/Parietal-INRIA/hidimstat'
DOWNLOAD_URL = 'https://github.com/Parietal-INRIA/hidimstat'
LICENSE = 'BSD'


def load_version():
    """Executes hidimstat/version.py in a globals dictionary and return it.
    Following format from Nilearn repo on github.
    """
    # load all vars into globals, otherwise
    #   the later function call using global vars doesn't work.
    globals_dict = {}
    with open(os.path.join('hidimstat', 'version.py')) as fp:
        exec(fp.read(), globals_dict)

    return globals_dict


def setup_package(version):
    local_path = os.path.dirname(os.path.abspath(sys.argv[0]))

    os.chdir(local_path)
    sys.path.insert(0, local_path)

    from numpy.distutils.core import setup

    setup(
        packages=find_packages(exclude=['contrib', 'docs', 'tests']),
        name=PKG,
        maintainer=MAINTAINER,
        include_package_data=True,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        long_description_content_type='text/markdown',
        license=LICENSE,
        url=URL,
        version=version,
        # download_url=DOWNLOAD_URL,
        zip_safe=False,  # the package can run out of an .egg file
        classifiers=[
            'Programming Language :: Python',
            'Programming Language :: Python :: 3.5',
            'Development Status :: 3 - Alpha'
        ],
    )


_VERSION_GLOBALS = load_version()
VERSION = _VERSION_GLOBALS['__version__']

if __name__ == "__main__":
    setup_package(VERSION)
