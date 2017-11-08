"""
Crop an image around an ellipse
"""
import numpy as np
from btracker.tools.geometry import Ellipse


def crop(self, image, ellipses,
         xmargin=[0, 0], ymargin=[0, 0]):
    if not isinstance(image, np.ndarray):
        raise TypeError('Image should be a ndarray')
    if not image.ndim >= 2:
        raise TypeError('Image should have at least two dimension')
    croped_images = list()
    for i, ell in enumerate(ellipses):
        if not isinstance(ell, Ellipse):
            raise TypeError(
                'Element #{} in ellipses is not of type Ellipse').format(i)
        y_spread, x_spread = ell.spread()
        y_range = np.arange(np.floor(ell.x - y_spread / 2)
                            - ymargin[0],
                            np.ceil(ell.x + y_spread / 2)
                            + ymargin[1])
        y_range = y_range.astype(np.int)
        x_range = np.arange(np.floor(ell.y - x_spread / 2)
                            - xmargin[0],
                            np.ceil(ell.y + x_spread / 2)
                            + xmargin[1])
        x_range = x_range.astype(np.int)
        y_range = np.clip(y_range, 0, image.shape[1] - 1)
        x_range = np.clip(x_range, 0, image.shape[0] - 1)
        croped_im = image[x_range, ...]
        croped_im = croped_im[:, y_range, ...]
        croped_images.append(croped_im)
    return croped_images
