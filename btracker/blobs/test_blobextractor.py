import unittest
from btracker.blobs.finder import BlobFinder
from btracker.tools.generator import background,mask,randomblob
import cv2
import numpy as np


class TestBlobFinder(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestBlobFinder, self).__init__(*args, **kwargs)
        self.run_ntimes_oneblob()

    def run_ntimes_oneblob(self):
        """
        Run the BlobFinder on one blob
        """
        self.background = background()
        self.mask = mask(self.background)
        blob = randomblob(self.background)[0]
        self.bfinder = BlobFinder(self.mask)
        # init background
        for _ in range(10):
            blob = randomblob(self.background)[0]
            self.bfinder.run(blob)

        self.blobs_center = list()
        self.blobs_found = list()
        for frame_i in range(10):
            blob, blob_center, _, _ = \
                randomblob(self.background)
            self.bfinder.run(blob)
            self.blobs_center.append(blob_center)
            self.blobs_found.append(self.bfinder.filtered_contours)

        self.blob = blob

    def test_setget_image(self):
        myim = self.bfinder.original_image
        self.assertTrue(np.allclose(myim, self.blob))

    def test_masked(self):
        myim = self.bfinder.masked_image
        testimage = cv2.bitwise_and(self.blob, self.blob, mask=self.mask)
        self.assertTrue(np.allclose(myim, testimage))

    def test_segmented(self):
        myim = self.bfinder.segmented_image
        self.assertTrue(np.allclose(np.unique(myim), [0, 255]))

    def test_oneblob_numbers(self):
        condition = True
        for frame_i in range(len(self.blobs_center)):
            contours = self.blobs_found[frame_i]
            if len(contours) > 1:
                print('Too many blobs detected')
                condition = False
                continue
            elif len(contours) < 1:
                print('Too few blobs detected')
                condition = False
                continue
        self.assertTrue(condition)

    def test_oneblob_positions(self):
        condition = True
        for frame_i in range(len(self.blobs_center)):
            contours = self.blobs_found[frame_i]
            contours = contours[0]
            px_error = np.sqrt(np.sum(contours.center
                                      - self.blobs_center[frame_i])**2)
            if px_error > 1:
                print('Pixel error too large : {}'.format(px_error))
                condition = False
        self.assertTrue(condition)


if __name__ == '__main__':
    unittest.main()
