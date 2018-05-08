"""
  A window
"""
import cv2
import numpy as np
from btracker.blobs.finder import BlobFinder
import btracker.tools.generator as btgen
from btracker.plot.image import overlay_ellipses
import yaml
import glob


def load_config(config_yaml):
    with open(config_yaml, 'r') as stream:
        try:
            config = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return config


def update_config_bfinder(bfinder, conf):
    """
    Update the bfinder with a dictionary
    for configuration
    ..todo: add checker
    """
    bfinder.erode_iter = conf['erode_iter']
    bfinder.dilate_iter = conf['dilate_iter']
    bfinder.area_lim = conf['area_lim']
    bfinder.roundness_lim = conf['roundness_lim']
    bfinder.background_init = conf['background_init']
    bfinder.threshold = conf['threshold']
    bfinder.gaussian_blur = conf['gaussian_blur']


template_frame = None
state = 'framebyframe'
config_yaml = 'tutorials/windowed_tracking.yaml'
scale = 0.3

if template_frame is None:
    # mask
    background = btgen.background()
    mask = btgen.mask(background)
    blob_brightness = [20, 30, 150, 254]
else:
    filelist = sorted(glob.glob(template_frame))

# Create a bfinder tracker
bfinder = BlobFinder(mask)
config = load_config(config_yaml)
update_config_bfinder(bfinder, config['bfinder'])
bfinder.skip_filter_contours = False


frame_i = 0

while True:
    # Reload params to allow an interactive way of
    # using the script
    config = load_config(config_yaml)
    update_config_bfinder(bfinder, config['bfinder'])

    # grab the current frame and initialize the occupied/unoccupied
    frame = btgen.randomblob(background,
                             color=blob_brightness[0])[0]
    for blobi in range(1, len(blob_brightness)):
        bcolor = blob_brightness[blobi]
        frame = btgen.randomblob(frame, color=bcolor)[0]
    bfinder.run(frame)
    # Get images for plot
    masked_image = bfinder.masked_image.copy()
    blured_image = bfinder.blured_image.copy()
    segmented_image = bfinder.segmented_image.copy()
    thresholded_image = bfinder.thresholded_image.copy()
    dilated_image = bfinder.dilated_image.copy()
    eroded_image = bfinder.eroded_image.copy()
    ellipses = bfinder.filtered_contours
    overlay_ellipses(ellipses, masked_image, firstn=len(ellipses))

    # Resize and concat image for display
    masked_image = cv2.resize(masked_image, None,
                              fx=scale, fy=scale,
                              interpolation=cv2.INTER_NEAREST)
    blured_image = cv2.resize(blured_image, None,
                              fx=scale, fy=scale,
                              interpolation=cv2.INTER_NEAREST)
    segmented_image = cv2.resize(segmented_image, None,
                                 fx=scale, fy=scale,
                                 interpolation=cv2.INTER_NEAREST)
    thresholded_image = cv2.resize(thresholded_image, None,
                                   fx=scale, fy=scale,
                                   interpolation=cv2.INTER_NEAREST)
    dilated_image = cv2.resize(dilated_image, None,
                               fx=scale, fy=scale,
                               interpolation=cv2.INTER_NEAREST)
    eroded_image = cv2.resize(eroded_image, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_NEAREST)
    todisplay = np.vstack(
        [np.hstack([masked_image, blured_image, segmented_image]),
         np.hstack([thresholded_image, dilated_image, eroded_image])])

    cv2.imshow('Interactive Tracker', todisplay)

    # -- Start of the interface --
    if state == 'runforward':
        frame_i += 1
        key = cv2.waitKey(10)
    elif state == 'framebyframe':
        key = cv2.waitKey()
    else:
        raise NameError('state unknown')
    if key == ord('q'):  # Quit
        print('Quit')
        break
    elif key == ord('f'):  # Forward by  one frame
        state = 'framebyframe'
        frame_i += 1
    elif key == ord('b'):  # Backward by one frame
        state = 'framebyframe'
        frame_i -= 1
    elif key == ord('r'):  # Run until stop or end
        state = 'runforward'
    elif key == ord('s'):  # Stop and switch to frame by frame
        state = 'framebyframe'
    print(frame_i)


cv2.destroyAllWindows()
