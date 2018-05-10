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
import btracker.io.ivfile as btio
import os
import pandas as pd
import time
from btracker.blobs.croped import crop

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
    bfinder.dilate_iter_first = conf['dilate_iter_first']
    bfinder.area_lim = conf['area_lim']
    bfinder.roundness_lim = conf['roundness_lim']
    bfinder.background_init = conf['background_init']
    bfinder.background_learning_rate = conf['background_learning_rate']
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
                        default='trajectory.hdf',
                        help=arghelp)
    arghelp = 'Scale to display images'
    parser.add_argument('--scale',
                        type=float,
                        default=0.3,
                        help=arghelp)
    return parser


args = parser_tracker().parse_args()
template_frame = args.images
state = 'runforward'
config_yaml =  args.config
scale =  args.scale
# Parameter for croping
xmargin = [0, 0]
ymargin = [0, 0]

if template_frame is None:
    # mask
    background = btgen.background()
    mask = btgen.mask(background)
    blob_brightness = [20, 30, 150, 254]
    bfinder = BlobFinder(mask)
else:
    filelist = sorted(glob.glob(template_frame))
    bfinder = BlobFinder(None, fgbg=cv2.createBackgroundSubtractorKNN())

# Create a bfinder tracker
config = load_config(config_yaml)
update_config_bfinder(bfinder, config['bfinder'])
bfinder.skip_filter_contours = False

# init background
maxframe = len(filelist)
print('INIT Background')
if template_frame is not None:
    bfinder.background_learning_rate = -1
    for frame_i in range(bfinder.background_init,0,-1):
        frame = cv2.imread(filelist[frame_i], 0)  # 0 for blackwhite
    bfinder.run(frame)
update_config_bfinder(bfinder, config['bfinder'])
    
# Create a dataframe
ellipses_all = pd.DataFrame(index=range(maxframe),
                            columns=[])



frame_i = 0
while True:
    # Check that frame number is not above last one
    # and previous one
    if frame_i >= (maxframe):
        break
    elif frame_i<0:
        frame_i=0
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
    croped_images = crop(masked_image, bfinder.filtered_contours, xmargin, ymargin)
    scores = []
    for cropim, _,_ in croped_images:
        cscore = cropim.max()-cropim.min()
        scores.append(cscore)
    if len(croped_images)>0:
        argsort = np.argsort(scores).astype(int)[::-1]
        ellipses=np.array(bfinder.filtered_contours)[argsort]
        ellipses=ellipses.tolist()
    else:
        ellipses=[]
    
    if len(ellipses)>0:
        columns = []
        for marki in range(0, len(ellipses)):
            for coli in btio.get_ellipse_param():
                columns.append((marki,coli))
        ellipse_series = pd.Series(index=pd.MultiIndex.from_tuples(columns))
        for marki, ell in enumerate(ellipses):
            ellipse_series.loc[(marki,'x')]=ell.x
            ellipse_series.loc[(marki,'y')]=ell.y
            ellipse_series.loc[(marki,'orientation')]=ell.angle
            ellipse_series.loc[(marki,'size')]=ell.area
            ellipse_series.loc[(marki,'roundness')]=ell.roundness
        ellipse_series.name=frame_i        
        if ellipse_series.name in ellipses_all.index:
            ellipses_all=ellipses_all.drop(ellipse_series.name)
        ellipses_all=ellipses_all.append(ellipse_series)
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

#Saving
print('Saving')
mcolumns=pd.MultiIndex.from_tuples(ellipses_all.columns.values)
mcolumns.rename('marker',level=0, inplace=True)
mcolumns.rename('ellipse_param',level=1, inplace=True)
ellipses_all.columns=mcolumns
ellipses_all.sort_index(inplace=True)
_, ext = os.path.splitext(args.tra_file)
if ext == '.hdf':
    ellipses_all.to_hdf(args.tra_file, key='tracking')
else:
    btio.save(args.tra_file, ellipses_all)

cv2.destroyAllWindows()
