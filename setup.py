#!/usr/bin/env python
""" setup.py for 3d reconstruction and calibration
"""
from setuptools import setup, find_packages

excluded = []


def exclude_package(pkg):
    for exclude in excluded:
        if pkg.startswith(exclude):
            return True
    return False


def create_package_list(base_package):
    return ([base_package] +
            [base_package + '.' + pkg
             for pkg
             in find_packages(base_package)
             if not exclude_package(pkg)])


setup_dict = {'name': 'btracker',
              'version': '0.1',
              'author': "Olivier J.N. Bertrand",
              'author_email': 'olivier.bertrand@uni-bielefeld.de',
              'description': 'Camera calibration and 3d reconstruction',
              'packages': create_package_list("btracker"),
              'requires': ['numpy', 'pandas', 'matplotlib', 'opencv'],
              'install_requires': ["numpy", 'pandas', 'matplotlib',
                                   'sphinx_rtd_theme','opencv-python']}

setup(**setup_dict)
