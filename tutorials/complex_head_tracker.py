
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
    print('Calculate background')
    # grab the current frame and initialize the occupied/unoccupied
    if maxframe is None:
        maxframe = camera.get(cv2.CAP_PROP_FRAME_COUNT)
    frames = np.floor(np.linspace(0,maxframe-1, nframe)).astype(int)
    #(grabbed, background) = camera.read()
    #background*=0
    #background=background.astype(float)
    background = cv2.createBackgroundSubtractorKNN(history = nframe)
    for fi in frames:
        camera.set(cv2.CAP_PROP_POS_FRAMES,fi)
        (grabbed, bg) = camera.read()
        bg = bg[...,0]
        #background+=bg/nframe_bg
        background.apply(bg)
    print('End background')
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
                    default=np.inf,    help="refresh time")

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
            args["tra_file"] = os.path.join(folder_path, 'trajectory.tra')
    display=False
    bbee = BlobFinder(mask, None)
    bbee.erode_iter = 2
    bbee.dilate_iter = 3
    bbee.threshold = 128
    bbee.gaussian_blur = 3
    bbee.skip_filter_contours = True

    #Parameters markers:
    max_area  = 40
    params = [80,300]
    gaussian_blur=5
    
    # Create image for display
    scale = 0.4
    refresh_time = 0.5
    nframe_bg = 1000
 
    camera = cv2.VideoCapture(argcam)
    background = get_background(camera, nframe_bg)
    #background = background[...,0]
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

        # Apply background
        frame=frame[...,0]
        diff = background.apply(frame.copy(),
                                learningRate=0)#frame-background
        #Find largest contours
        bbee.run(diff)
        area  = 0
        cont = []
        for con in bbee.contours:
            moments = cv2.moments(con)
            if moments['m00'] <=0:
                continue
            if moments['m00']>area:
                area = moments['m00']
                cont = con
                cx_bee = moments['m10']/moments['m00']
                cy_bee = moments['m01']/moments['m00']
        
        if len(cont) < 5:
            continue
        ell = Ellipse()
        ell.from_opencv_tuples(cv2.fitEllipse(cont))
        ell.width = 1.5* ell.width
        ell.height = 1.5* ell.height
        # Crop bee
        mask = diff.copy()*0
        cv2.drawContours(mask, [cont], -1,
                             255, -1) 
        bee_im=frame.copy()
        bee_im[mask<128]=0
        (bee_im, bee_im_x, bee_im_y) = btcrop.crop(bee_im,[ell])[0]

        
        blurred=cv2.GaussianBlur(bee_im,
                                 (gaussian_blur,gaussian_blur),0)

        
        edges = cv2.Canny(blurred,params[0],params[1])
        edges = cv2.dilate(edges, None,iterations=2)
        edges = cv2.erode(edges, None,iterations=2)

        _, contours, _ = cv2.findContours(edges,
                                          cv2.RETR_TREE,
                                          cv2.CHAIN_APPROX_SIMPLE)
        contours_filtered= list()
        ellipses = list()
        for con in contours:
            moments = cv2.moments(con)
            if moments['m00'] <=0:
                continue
            cx = moments['m10']/moments['m00']
            cy = moments['m01']/moments['m00']
            if moments['m00']<=max_area:
                contours_filtered.append(con)
                ellipses.append(Ellipse(x=cx+bee_im_x,
                                        y=cy+bee_im_y,
                                        width = 3,
                                        height = 3))
        btio.append(file, frame_i, ellipses)
        # display images when asked
        if display:
            if (time.time() - t_start) > refresh_time:
                t_start = time.time()
                toplot = frame.copy()
                toplot = bfimshow.bw2color(toplot)
                bfimshow.overlay_ellipses([ell], toplot, np.inf)
                bfimshow.overlay_ellipses(ellipses, toplot, np.inf)
                cv2.drawContours(toplot, cont, -1, (0,255,0), 3)
                bfimshow.draw(toplot, 'bla', scale, frame_i)
                toplot2 = bee_im.copy()
                toplot2 = np.hstack([toplot2,
                                 edges])
                toplot2=bfimshow.bw2color(toplot2)
                cv2.drawContours(toplot2, contours_filtered, -1,
                             (0,255,0), 3)
            
                cv2.imshow('bee', toplot2)
    # Before next loop run
    file.close()
    # release cam
    camera.release()

    # cleanup the camera and close any open windows
    cv2.destroyAllWindows()
