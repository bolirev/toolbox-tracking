#!/usr/bin/env python
"""
setup.py for toolbox-tracking

for install it needs:
pip install opencv-python
pip install matplotlib
pip install pandas
pip install scipy

sudo apt-get install libblas-dev liblapack-dev libatlas-base-dev gfortran
sudo apt-get install libboost-all-dev
python install setup.py

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


setup_info = dict(name='btracker',
                  version='0.1',
                  author="Olivier J.N. Bertrand",
                  description='Tracking of Blobs, like bees or flies',
                  packages=create_package_list('btracker'),
                  requires=['numpy', 'cv2', 'pandas', 'scipy'],
                  install_requires=["numpy", 'pandas', 'scipy'])

setup(**setup_info)
