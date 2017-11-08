import cv2
import numpy as np

def background():
    backim = (np.random.rand(1080, 1980) * 100)
    backim = backim.astype(np.uint8) + 128
    return backim

def mask(backim):
    maskim = backim.copy()
    maskim[:] = 0
    cv2.circle(maskim, (maskim.shape[1] // 2,
                      maskim.shape[0] // 2),
               maskim.shape[0] // 2, (255, 0, 0), -1)
    return maskim

def randomblob(background_im,color=20):
    blob = background_im.copy()
    blob_max_radius = background_im.shape[0] // 20
    blob_min_radius = 5
    blob_center = (np.random.rand(2) - 0.5) *\
            (background_im.shape[0] // 2 - blob_max_radius)
    blob_center[0] += background_im.shape[1] // 2
    blob_center[1] += background_im.shape[0] // 2
    blob_center = blob_center.astype(np.uint)
    blob_axes = np.random.rand(2) * (blob_max_radius - blob_min_radius)
    blob_axes += blob_min_radius
    blob_axes = blob_axes.astype(np.uint)
    blob_axes.sort()
    blob_angle = np.random.rand() * 180
    cv2.ellipse(blob,
                (blob_center[0], blob_center[1]),
                (blob_axes[0], blob_axes[1]),
                blob_angle, 0, 360, color, -1)
    return blob, blob_center, blob_axes[::-1] * 2, blob_angle
