"""


"""
# import the necessary packages
import time
import cv2
from btracker.blobs.finder import BlobFinder
from btracker.blobs.croped import crop
import btracker.plot.image as bfimshow
import btracker.tools.generator as btgen


def draw(croped_images, max_nb_images):
    for i in range(min(len(croped_images), max_nb_images)):
        cropim = croped_images[i]
        windowname = 'croped image {}'.format(i)
        if bfimshow.draw(cropim, windowname, scale=1, frame_i=None):
            return True
    return False


# Create image for display
scale = 0.4
refresh_time = 0.5
nblobs = 2
max_nb_images = 2
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
# Parameter for croping
xmargin = [0, 0]
ymargin = [0, 0]

frame_i = 0
while True:
    # grab the current frame and initialize the occupied/unoccupied
    frame = btgen.randomblob(background, color=100)[0]
    for _ in range(1, nblobs):
        frame = btgen.randomblob(frame, color=100)[0]
    bfinder.run(frame)
    #
    image = bfinder.original_image
    image[bfinder.processed_image == 0] = 255
    croped_images = crop(image, bfinder.filtered_contours, xmargin, ymargin)
    # display images when asked
    if (time.time() - t_start) > refresh_time:
        t_start = time.time()
        if draw(croped_images, max_nb_images):
            break
    # Before next loop run
    frame_i += 1

# cleanup the camera and close any open windows
cv2.destroyAllWindows()
