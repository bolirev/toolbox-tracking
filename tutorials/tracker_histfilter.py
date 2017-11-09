"""


"""
# import the necessary packages
import time
import cv2
import numpy as np
from btracker.blobs.finder import BlobFinder
from btracker.blobs.croped import crop
import btracker.plot.image as bfimshow
import btracker.tools.generator as btgen
import btracker.blobs.filter as btfilt


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
blob_brightness = [20, 30, 150, 254]
max_nb_images = 1
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

# Build up an expected histogram
ninitframes = 50
bin_edges = np.arange(1, 255)
ref_cumsum = np.zeros(bin_edges.shape[0] - 1)
for _ in range(bfinder.background_init):
    # grab the current frame and initialize the occupied/unoccupied
    frame = btgen.randomblob(background,
                             color=blob_brightness[0])[0]
    bfinder.run(frame)

for _ in range(ninitframes):
    # grab the current frame and initialize the occupied/unoccupied
    frame = btgen.randomblob(background,
                             color=blob_brightness[0])[0]
    bfinder.run(frame)
    #
    image = bfinder.original_image
    image = image.astype(np.float)
    image[bfinder.processed_image == 0] = np.nan
    croped_images = crop(image, bfinder.filtered_contours, xmargin, ymargin)
    hist = btfilt.image2hist(croped_images[0], bin_edges)
    cumsum = btfilt.hist2cumsum(hist)
    ref_cumsum += cumsum / ninitframes

frame_i = 0
while True:
    # grab the current frame and initialize the occupied/unoccupied
    frame = btgen.randomblob(background,
                             color=blob_brightness[0])[0]
    for blobi in range(1, len(blob_brightness)):
        bcolor = blob_brightness[blobi]
        frame = btgen.randomblob(frame, color=bcolor)[0]
    bfinder.run(frame)
    #
    image = bfinder.original_image.copy()
    image[bfinder.processed_image == 0] = 255
    croped_images = crop(image, bfinder.filtered_contours, xmargin, ymargin)
    if len(croped_images) <= 0:
        continue
    scores = btfilt.score_cumsum_hist(croped_images, ref_cumsum, bin_edges)
    croped_images = [croped_images[np.argmin(scores)]]
    ellipse = [bfinder.filtered_contours[np.argmin(scores)]]
    # Filter croped images

    # display images when asked
    if (time.time() - t_start) > refresh_time:
        t_start = time.time()
        if draw(croped_images, max_nb_images):
            break
        if bfimshow.original_image(bfinder, scale, frame_i, 5,
                                   ellipses=ellipse):
            break
    # Before next loop run
    frame_i += 1


# cleanup the camera and close any open windows
cv2.destroyAllWindows()
