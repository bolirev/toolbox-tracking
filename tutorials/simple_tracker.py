"""


"""
# import the necessary packages
import time
import cv2
from btracker.blobs.finder import BlobFinder
import btracker.plot.image as bfimshow
import btracker.tools.generator as btgen
# Create image for display
scale = 0.4
refresh_time = 0.5
nblobs = 2
# Bfinder parameters
args = dict()

# loop over the frames of the video
t_start = time.time()
# mask
background = btgen.background()
mask = btgen.mask(background)
# Create a bfinder tracker
bfinder = BlobFinder(mask)
bfinder.erode_iter = 2
bfinder.dilate_iter = 2
bfinder.area_lim = [0, 10000]
bfinder.roundness_lim = [0, 1]
bfinder.background_init = 60
bfinder.threshold = 10
bfinder.gaussian_blur = 1
bfinder.skip_filter_contours = False

frame_i = 0
while True:
    # grab the current frame and initialize the occupied/unoccupied
    frame = btgen.randomblob(background, color=100)[0]
    for _ in range(1, nblobs):
        frame = btgen.randomblob(frame, color=100)[0]
    bfinder.run(frame)
    # display images when asked
    if (time.time() - t_start) > refresh_time:
        t_start = time.time()
        if bfimshow.original_image(bfinder, scale, frame_i, 5):
            break
    # Before next loop run
    frame_i += 1

# cleanup the camera and close any open windows
cv2.destroyAllWindows()
