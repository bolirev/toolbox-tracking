"""
   Tool to click a manhattan
"""
import argparse
import cv2
import pandas as pd
import numpy as np
import os


parser = argparse.ArgumentParser()
arghelp = 'Path to the folder containing the data'
parser.add_argument('--file',
                    type=str,
                    default=None,
                    help=arghelp)

arghelp = 'Number of points on the manhattan'
parser.add_argument('--npoints',
                    type=int,
                    default=28,
                    help=arghelp)


args = parser.parse_args()
print(args)
# Check that the folder exists and the subfolders as well
# if not os.path.exists(args.file):
#    raise IOError('{} does not exist'.format(args.file)
points = pd.DataFrame(index=range(args.npoints),
                      columns=['x', 'y'])
points.name = 'manhattan_view'
points.unit = 'px'

img = cv2.imread(args.file)
img_target_size = (640, 480)
print(img.shape)


def scale_image(img, img_target_size):
    if img.shape[0] / img_target_size[0] > img.shape[1] / img_target_size[1]:
        scale = img_target_size[0] / img.shape[0]
    else:
        scale = img_target_size[1] / img.shape[1]
    img_resized = cv2.resize(img, (0, 0), fx=scale, fy=scale)
    return scale, img_resized


def coarse_zoom(event, x, y, flags, param):
    global ix, iy, mode
    if event == cv2.EVENT_MOUSEMOVE:
        ix = x
        iy = y
    if event == cv2.EVENT_LBUTTONDOWN:
        mode = 'next'
    if event == cv2.EVENT_MBUTTONDOWN:
        mode = 'skip'


def crop_image(img, center, size):
    x_range = np.arange(np.floor(center[0] - size[0] // 2),
                        np.floor(center[0] + size[0] // 2))
    y_range = np.arange(np.floor(center[1] - size[1] // 2),
                        np.floor(center[1] + size[1] // 2))
    x_range = x_range.astype(int)
    y_range = y_range.astype(int)
    x_range = np.clip(x_range, 0, img.shape[1] - 1)
    y_range = np.clip(y_range, 0, img.shape[0] - 1)
    croped_im = img[y_range, ...]
    croped_im = croped_im[:, x_range, ...]
    return croped_im


scale, img_resized = scale_image(img, img_target_size)
cv2.imshow(args.file, img_resized)
cv2.setMouseCallback(args.file, coarse_zoom)

zoomsize = (100, 100)
ix_old = 0
iy_old = 0
ix = img_target_size[0] // 2
iy = img_target_size[1] // 2
mode = 'move'
point_i = 0
while True:
    key = cv2.waitKey(1)
    if key == 113:
        break
    if (ix_old != ix) or (iy_old != iy):
        cimage = img_resized.copy()
        cv2.line(cimage,
                 (ix, 0),
                 (ix, cimage.shape[0]),
                 (0, 0, 255), 1)
        cv2.line(cimage,
                 (0, iy),
                 (cimage.shape[1], iy),
                 (0, 0, 255), 1)
        cv2.imshow(args.file, cimage)

        center = np.array([ix, iy])
        center = center.astype(float)
        center /= scale
        cimage = crop_image(img, center, size=zoomsize)
        _, cimage = scale_image(cimage, img_target_size)
        cv2.line(cimage,
                 (cimage.shape[1] // 2, 0),
                 (cimage.shape[1] // 2, cimage.shape[0]),
                 (0, 0, 255), 1)
        cv2.line(cimage,
                 (0, cimage.shape[0] // 2),
                 (cimage.shape[1], cimage.shape[0] // 2),
                 (0, 0, 255), 1)
        cv2.imshow('zoom', cimage)

    if mode == 'next':
        x = ix
        y = iy
        x /= scale
        y /= scale
        points.loc[point_i, :] = [x, y]
        point_i += 1
        mode = 'move'
        print(points)
        print('next point: {}'.format(point_i))
    elif mode == 'skip':
        point_i += 1
        mode = 'move'
        print(points)
        print('next point: {}'.format(point_i))
cv2.destroyAllWindows()

dirname = os.path.dirname(args.file)
basename = os.path.splitext(os.path.basename(args.file))[0]
points.to_hdf(os.path.join(dirname, basename + '_clicked.h5'), key='manhattan')
