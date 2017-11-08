"""


"""
# import the necessary packages
import argparse
import time
import cv2
from btracker.blobs.finder import BlobFinder
import bttracker.plots.image as bfimshow

# Create image for display
scale = 0.4


def process_images(camera, args):
    """
            process images from camera, and store the result in a file

            :param camera: a camera opencv stream
            :type frame: output of cv2.VideoCapture
            :param file: file to store the tracking results
            :type file: file object, output of open()
            :param args: parameters for the tracking
            :type args: dict

            ..seealso: process_image
    """
    # loop over the frames of the video
    t_start = time.time()
    frame_i = 0
    if args["mask"] is None:
        mask = None
    else:
        mask = cv2.imread(args["mask"], 0)
    # Create a bfinder tracker
    bfinder = BlobFinder(mask)
    for key, val in args:
        setattr(bfinder, key, val)
    while True:
        # grab the current frame and initialize the occupied/unoccupied
        (grabbed, frame) = camera.read()
        # if the frame could not be grabbed, then we have reached the end
        # of the video
        if not grabbed:
            break
        bfinder.run(frame)
        # display images when asked
        if (time.time() - t_start) > args["refresh_time"]:
            t_start = time.time()
            if bfimshow.original_image(bfinder, scale, frame_i, 5):
                break
        # Before next loop run
        frame_i += 1

    # cleanup the camera and close any open windows
    camera.release()
    cv2.destroyAllWindows()


def process(args):
    """	Track bees on a video
            :param args: parameters
            :type args: dict()

            args key explanation run: simpleTracker -h
    """
    # if the video argument is None, then we are reading from webcam
    if (args.get("video", None) is None) and \
            (args.get("folder", None) is None):
        camera = cv2.VideoCapture(0)
        time.sleep(0.25)
    # Load image for file list if video is None
    elif args.get("video", None) is None:
        camera = cv2.VideoCapture(args["folder"])

    # otherwise, we are reading from a video file
    else:
        camera = cv2.VideoCapture(args["video"])

    process_images(camera, args)
    # release cam
    camera.release()


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
    ap.add_argument("-f", "--tra-file",  default=None,
                    help="trajectory file e.g. /fullpath/trajectory.tra to store the tracking result")
    args = vars(ap.parse_args())
    # Add unpassable parameters
    args["gblur"] = 21
    args["dilate_iter"] = 2
    args["erode_iter"] = 2
    process(args)
