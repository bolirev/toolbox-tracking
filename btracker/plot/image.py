"""

"""
import cv2
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from btracker.blobs.finder import BlobFinder


def overlay_ellipses(ellipses, image, firstn,
                     cmap=plt.get_cmap('hsv'),
                     linewidth=2):
    # overlay ellipses
    norm = mpl.colors.Normalize(vmin=0, vmax=len(ellipses))
    scalar_map = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    for i, ell in enumerate(ellipses):
        if i > firstn:
            break
        colour = scalar_map.to_rgba(i)
        colour_bgr = np.array(colour[2::-1]) * 255
        cv2.ellipse(image, (ell.center, ell.size, ell.angle),
                    colour_bgr, linewidth)


def draw(image, windowname, scale, frame_i):
    todisp = cv2.resize(image, None, fx=scale, fy=scale,
                        interpolation=cv2.INTER_NEAREST)
    # Addtext on image
    if frame_i is not None:
        cv2.putText(todisp, str(frame_i), (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow(windowname, todisp)

    key = cv2.waitKey(1) & 0xFF

    # if the `q` key is pressed, break from the lop
    if key == ord("q"):
        return True
    else:
        return False


def bw2color(image):
    image = image[..., np.newaxis]
    image = np.repeat(image, 3, axis=2)
    return image


def original_image(bfinder, scale=0.2,
                   frame_i=None, overlay_firstn_ellipses=-1,
                   ellipses=None):
    if ellipses is None:
        ellipses = bfinder.filtered_contours
    if not isinstance(bfinder, BlobFinder):
        raise TypeError('bfinder should be BlobFinder type')
    im2display = bw2color(bfinder.original_image)
    windowname = 'original_image'
    overlay_ellipses(ellipses, im2display, overlay_firstn_ellipses)
    return draw(im2display, windowname, scale, frame_i)


def masked_image(bfinder, scale=0.2,
                 frame_i=None, overlay_firstn_ellipses=-1,
                 ellipses=None):
    if ellipses is None:
        ellipses = bfinder.filtered_contours
    if not isinstance(bfinder, BlobFinder):
        raise TypeError('bfinder should be BlobFinder type')
    im2display = bfinder.masked_image
    windowname = 'masked_image'
    overlay_ellipses(ellipses, im2display, overlay_firstn_ellipses)
    return draw(im2display, windowname, scale, frame_i)


def processed_image(bfinder, scale=0.2,
                    frame_i=None, overlay_firstn_ellipses=-1,
                    ellipses=None):
    if ellipses is None:
        ellipses = bfinder.filtered_contours
    if not isinstance(bfinder, BlobFinder):
        raise TypeError('bfinder should be BlobFinder type')
    im2display = bw2color(bfinder.processed_image)
    windowname = 'processed_image'
    overlay_ellipses(ellipses, im2display, overlay_firstn_ellipses)
    return draw(im2display, windowname, scale, frame_i)
