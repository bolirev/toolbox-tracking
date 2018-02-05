"""
    simple Tracker is a small python script based on btracker
"""
# import the necessary packages
import argparse
import time
import cv2
import numpy as np
import os
from btracker.blobs.finder import BlobFinder
import btracker.blobs.croped as btcrop
import btracker.plot.image as bfimshow
import btracker.blobs.filter as btfilt
import btracker.io.load_write_2d_tra_file as btio
from btracker.tools.geometry import Ellipse
import glob


#
# Create image for display
scale = 0.4
refresh_time = np.inf
score_th = 0.7

# loop over the frames of the video
t_start = time.time()
# mask

def get_background(camera, nframe, maxframe=None):
    """ Calculate background from nframe
        by acumulation of images
    """
    # grab the current frame and initialize the occupied/unoccupied
    if maxframe is None:
        maxframe = camera.get(cv2.CAP_PROP_FRAME_COUNT)
    frames = np.floor(np.linspace(0,maxframe-1, nframe)).astype(int)
    (grabbed, background) = camera.read()
    background*=0
    background=background.astype(float)
    for fi in frames:
        camera.set(cv2.CAP_PROP_POS_FRAMES,fi)
        (grabbed, bg) = camera.read()
        background+=bg/nframe_bg
    return background


if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video",
                    default=None, help="path to the video file")
    ap.add_argument("-m", "--mask",  default=None,
                    help="filename for image mask e.g. /fullpath/my_mask.jpg")
    ap.add_argument("-d", "--folder",                default=None,
                    help="folder template for image list (e.g. /fullpath/top_(percent)05d.jpg)")
    ap.add_argument("-t", "--threshold",   type=int,
                    default=128,   help="threshold (0 to 255)")
    ap.add_argument("-f", "--tra-file",  default=None,
                    help="trajectory file e.g. /fullpath/trajectory.tra to store the tracking result")
    ap.add_argument("-r", "--refresh-time",
                    default=1,    help="refresh time")

    args = vars(ap.parse_args())
    # Load mask
    if args["mask"] is None:
        mask = None
    else:
        mask = cv2.imread(args["mask"], 0)

    # if the video argument is None, then we are reading from webcam
    if (args.get("video", None) is None) and (args.get("folder", None) is None):
        camera = cv2.VideoCapture(0)
        time.sleep(0.25)
    # Load image for file list if video is None
    elif args.get("video", None) is None:
        argcam= args["folder"]
    # otherwise, we are reading from a video file
    else:
        argcam= args["video"]

    if args["tra_file"] is None:
        if args["folder"] is None:
            raise NameError('target trajectory file is not present')
        else:
            folder_path = os.path.dirname(args["folder"])
            args["tra_file"] = os.path.join(folder_path, 'trajectory_v2.tra')

    bmarker = BlobFinder(mask, None)
    bmarker.erode_iter = 1
    bmarker.dilate_iter = 1
    bmarker.area_lim = [0, 2000]
    bmarker.roundness_lim = [0, 1]
    bmarker.threshold = args["threshold"]
    bmarker.gaussian_blur = 1
    bmarker.skip_filter_contours = True

    # Create image for display
    scale = 0.4
    refresh_time = 0.5
    nframe_bg = 100
 
    camera = cv2.VideoCapture(argcam) 
    background = get_background(camera, nframe_bg)
    background = background[...,0]
    camera.release()
    
    camera = cv2.VideoCapture(argcam)

    # Process image
    file = open(args["tra_file"], 'w')
    frame_i = -1
    while True:
        frame_i += 1
        # grab the current frame and initialize the occupied/unoccupied
        (grabbed, frame) = camera.read()
        
        # if the frame could not be grabbed, then we have reached the end
        # of the video
        if not grabbed:
            break

        # Calculate normed difference
        frame=frame[...,0]
        diff = frame-background
        diff =np.abs(diff)
        diff-=diff.min()
        diff/=diff.max()
        diff*=255
        diff =255-diff.astype(np.uint8)
        # for display
        image =bfimshow.bw2color(diff.copy())

        #Find contours
        bmarker.run(diff)
        
        # Assign center of mass ellipses
        ellipses = list()
        for con in  bmarker.contours:
            moments = cv2.moments(con)
            if moments['m00'] <=0:
                continue
            cx = moments['m10']/moments['m00']
            cy = moments['m01']/moments['m00']
            ellipse = Ellipse(x=cx,y=cy, height=5, width=5)                
            ellipses.append(ellipse)
        btio.append(file, frame_i, ellipses)

        # display images when asked
        if (time.time() - t_start) > refresh_time:
            t_start = time.time()
            bfimshow.overlay_ellipses(ellipses, image, np.inf)
            bfimshow.draw(image, 'bla', scale, frame_i)
    # Before next loop run
    file.close()
    # release cam
    camera.release()

    # cleanup the camera and close any open windows
    cv2.destroyAllWindows()
