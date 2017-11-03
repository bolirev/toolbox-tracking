#!/usr/bin/env python
"""
setup.py for toolbox-tracking

for install it needs:
pip install opencv-python
pip install matplotlib
pip install pandas
pip install scipy
python install setup.py

"""
import sys
from setuptools import setup, find_packages, Extension


try:
    import numpy as np
except ImportError:
    print("Numpy must be installed before running setup.py.")
    if sys.version_info.major == 2:
        def input(*args, **kwargs):
            return raw_input(*args, **kwargs)
    response = input("Attempt to automatically install numpy using pip? (y/n)")
    if 'y' in response.lower():
        import subprocess
        subprocess.call(["pip", "install", "numpy"])
        import numpy as np
    else:
        raise ImportError("Numpy required before installation.")


my_module = Extension('beetracktracker',
                      # would it be better to link with the shared lib?
                      sources=['src/cpp/opencv_swig_python.cpp',
                               'src/cpp/BeeTrackTracker.cpp'],
                      include_dirs=['', np.get_include()],
                      library_dirs=['/usr/local/share/OpenCV/3rdparty/lib/'],
                      libraries=["boost_python-py35",
                                 "opencv_stitching",
                                 "opencv_superres",
                                 "opencv_videostab",
                                 "opencv_aruco",
                                 "opencv_bgsegm",
                                 "opencv_bioinspired",
                                 "opencv_ccalib",
                                 "opencv_dnn",
                                 "opencv_dpm",
                                 "opencv_fuzzy",
                                 "opencv_hdf",
                                 "opencv_line_descriptor",
                                 "opencv_optflow",
                                 "opencv_plot",
                                 "opencv_reg",
                                 "opencv_saliency",
                                 "opencv_stereo",
                                 "opencv_structured_light",
                                 "opencv_rgbd",
                                 "opencv_surface_matching",
                                 "opencv_tracking",
                                 "opencv_datasets",
                                 "opencv_text",
                                 "opencv_face",
                                 "opencv_xfeatures2d",
                                 "opencv_shape",
                                 "opencv_video",
                                 "opencv_ximgproc",
                                 "opencv_calib3d",
                                 "opencv_features2d",
                                 "opencv_flann",
                                 "opencv_xobjdetect",
                                 "opencv_objdetect",
                                 "opencv_ml",
                                 "opencv_xphoto",
                                 "ippicv",
                                 "opencv_highgui",
                                 "opencv_videoio",
                                 "opencv_imgcodecs",
                                 "opencv_photo",
                                 "opencv_imgproc",
                                 "opencv_core"])


setup(name='beetracktracker',
      version='0.1',
      author="Olivier J.N. Bertrand",
      description='Tracking of Bees',
      ext_modules=[my_module],
      py_modules=["beetracktracker"],
      packages=find_packages(exclude=["docs"]),
      requires=['numpy', 'cv2'],
      install_requires=["numpy", "opencv-python"])
