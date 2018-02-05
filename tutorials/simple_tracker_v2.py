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

#
# Create image for display
scale = 0.4
refresh_time = np.inf
score_th = 0.7

# loop over the frames of the video
t_start = time.time()
# mask


if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video",
                    default=None, help="path to the video file")
    ap.add_argument("-g", "--min-area",    type=int, default=100,
                    help="minimum area size [in pixel]")
    ap.add_argument("-l", "--max-area",    type=int,
                    default=1000, help="maximum area size [in pixel]")
    ap.add_argument("-n", "--max-nbbee",    type=int,
                    default=1, help="maximum number bees")
    ap.add_argument("-m", "--mask",  default=None,
                    help="filename for image mask e.g. /fullpath/my_mask.jpg")
    ap.add_argument("-r", "--refresh-time",
                    default=1,    help="refresh time")
    ap.add_argument("-d", "--folder",                default=None,
                    help="folder template for image list (e.g. /fullpath/top_(percent)05d.jpg)")
    ap.add_argument("-t", "--threshold",   type=int,
                    default=25,   help="threshold (0 to 255)")
    ap.add_argument("-f", "--tra-file",  default=None,
                    help="trajectory file e.g. /fullpath/trajectory.tra to store the tracking result")
    ap.add_argument("-e", "--bin-edges-file",  default=None,
                    help="bin edges file to open bee template edges")
    ap.add_argument("-c", "--refcumsum-file",  default=None,
                    help="ref cumsum file to open bee template brightness cumsum")

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
        camera = cv2.VideoCapture(args["folder"])

    # otherwise, we are reading from a video file
    else:
        camera = cv2.VideoCapture(args["video"])

    if args["tra_file"] is None:
        if args["folder"] is None:
            raise NameError('target trajectory file is not present')
        else:
            folder_path = os.path.dirname(args["folder"])
            args["tra_file"] = os.path.join(folder_path, 'trajectory_v2.tra')

    # Define BlobFinder
    bfinder = BlobFinder(mask)
    bfinder.erode_iter = 2
    bfinder.dilate_iter = 2
    bfinder.area_lim = [args['min_area'], args['max_area']]
    bfinder.roundness_lim = [0, 1]
    bfinder.background_init = 60
    bfinder.threshold = 10
    bfinder.gaussian_blur = 21
    bfinder.skip_filter_contours = False

    # Load ref cumsum and bin edges
    if args["bin_edges_file"] is None:
        raise NameError('bin edges file is not present')
    else:
        bin_edges = np.load(args["bin_edges_file"])

    # Load ref cumsum and bin edges
    if args["refcumsum_file"] is None:
        raise NameError('ref cumsum file is not present')
    else:
        ref_cumsum = np.load(args["refcumsum_file"])

    # Process images
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
        frame=frame[...,0]
        bfinder.run(frame)
        croped_images = btcrop.crop(frame, bfinder.filtered_contours)
        if len(croped_images) <= 0:
            btio.append(file, frame_i, [])
            continue
        scores = btfilt.score_cumsum_hist(croped_images, ref_cumsum, bin_edges)
        scores = np.array(scores)
        scores_id = scores < score_th
        if np.any(scores_id == True):
            scores = scores[scores_id]
        else:
            btio.append(file, frame_i, [])
            continue

        sorted_arg = np.argsort(scores)
        if len(sorted_arg) > args['max_nbbee']:
            sorted_arg = sorted_arg[:args['max_nbbee']]
        print(scores)
        ellipses = list()
        for besti in sorted_arg:
            ellipses.append(bfinder.filtered_contours[besti])
        btio.append(file, frame_i, ellipses)
        # display images when asked
        if (time.time() - t_start) > refresh_time:
            t_start = time.time()
            if bfimshow.original_image(bfinder, scale, frame_i, 5,
                                       ellipses=ellipses):
                break

    file.close()
    # release cam
    camera.release()

    # cleanup the camera and close any open windows
    cv2.destroyAllWindows()
