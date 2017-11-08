"""

"""
import cv2
import numpy as np
from btracker.blobs.finder import BlobFinder


def overlay_ellipses(bfinder, image, firstn, color=(0, 0, 255),
                     linewidth=2):
    # overlay ellipses
    for i, ell in enumerate(bfinder.filtered_contours):
        if i > firstn:
            break
        cv2.ellipse(image, (ell.center, ell.size, ell.angle),
                    color, linewidth)


def draw(image, windowname, scale, frame_i):
    todisp = cv2.resize(image, None, fx=scale, fy=scale,
                        interpolation = cv2.INTER_NEAREST)
    # Addtext on image
    cv2.putText(todisp, str(frame_i), (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow(windowname, todisp)

    key = cv2.waitKey(1) & 0xFF

    # if the `q` key is pressed, break from the lop
    if key == ord("q"):
        return True
    else:
        return False

def bw2color(image):
    image=image[...,np.newaxis]
    image=np.repeat(image,3,axis=2)
    return image

def original_image(bfinder, scale=0.2,
                   frame_i=None, overlay_firstn_ellipses=-1):
    if not isinstance(bfinder, BlobFinder):
        raise TypeError('bfinder should be BlobFinder type')
    im2display = bw2color(bfinder.original_image)
    windowname = 'original_image'
    overlay_ellipses(bfinder, im2display, overlay_firstn_ellipses)
    return draw(im2display, windowname, scale, frame_i)


def masked_image(bfinder, scale=0.2,
                 frame_i=None, overlay_firstn_ellipses=-1):
    if not isinstance(bfinder, BlobFinder):
        raise TypeError('bfinder should be BlobFinder type')
    im2display = bfinder.masked_image
    windowname = 'masked_image'
    overlay_ellipses(bfinder, im2display, overlay_firstn_ellipses)
    return draw(im2display, windowname, scale, frame_i)


def processed_image(bfinder, scale=0.2,
                    frame_i=None, overlay_firstn_ellipses=-1):
    if not isinstance(bfinder, BlobFinder):
        raise TypeError('bfinder should be BlobFinder type')
    im2display = bfinder.processed_image
    windowname = 'processed_image'
    overlay_ellipses(bfinder, im2display, overlay_firstn_ellipses)
    return draw(im2display, windowname, scale, frame_i)
