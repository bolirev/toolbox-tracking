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
import argparse
import btracker.io.load_write_2d_tra_file as btio
import os


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

def parser_tracker():
    # Create command line options
    parser = argparse.ArgumentParser()
    arghelp = 'Template to the images (will be openned and sorted by glob)'
    parser.add_argument('--images',
                        type=str,
                        default=None,
                        help=arghelp)
    arghelp = 'Path to your yaml script for config'
    parser.add_argument('--config',
                        type=str,
                        default=None,
                        help=arghelp)
    arghelp = 'Path to trajectory'
    parser.add_argument('--tra-file',
                        type=str,
                        default='trajectory.tra',
                        help=arghelp)
    arghelp = 'Scale to display images'
    parser.add_argument('--scale',
                        type=float,
                        default=0.3,
                        help=arghelp)
    return parser


args = parser_tracker().parse_args()
template_frame = args.images
state = 'framebyframe'
config_yaml =  args.config
scale =  args.scale
trafile = open(args.tra_file, 'r+')

if template_frame is None:
    # mask
    background = btgen.background()
    mask = btgen.mask(background)
    blob_brightness = [20, 30, 150, 254]
    bfinder = BlobFinder(mask)
else:
    filelist = sorted(glob.glob(template_frame))
    bfinder = BlobFinder(None)

# Create a bfinder tracker
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
    if template_frame is None:
        frame = btgen.randomblob(background,
                             color=blob_brightness[0])[0]
    
        for blobi in range(1, len(blob_brightness)):
            bcolor = blob_brightness[blobi]
            frame = btgen.randomblob(frame, color=bcolor)[0]
    else:
        frame = cv2.imread(filelist[frame_i], 0)  # 0 for blackwhite
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
         np.hstack([thresholded_image, eroded_image, dilated_image])])

    cv2.imshow('Interactive Tracker', todisplay)
    btio.append(trafile, frame_i, ellipses)
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
        #Move the pointer (similar to a cursor in a text editor) to the end of the file. 
        trafile.seek(0, os.SEEK_END)

        #This code means the following code skips the very last character in the file - 
        #i.e. in the case the last line is null we delete the last line 
        #and the penultimate one
        pos = trafile.tell() - 1

        #Read each character in the file one at a time from the penultimate 
        #character going backwards, searching for a newline character
        #If we find a new line, exit the search
        while pos > 0 and trafile.read(1) != "\n":
            pos -= 1
            trafile.seek(pos, os.SEEK_SET)

            #So long as we're not at the start of the file, delete all the characters ahead of this position
            if pos > 0:
                trafile.seek(pos, os.SEEK_SET)
                trafile.truncate()
    elif key == ord('r'):  # Run until stop or end
        state = 'runforward'
    elif key == ord('s'):  # Stop and switch to frame by frame
        state = 'framebyframe'
    print(frame_i)
    trafile.flush()

trafile.close()
cv2.destroyAllWindows()
