import unittest
from btracker.blobs.finder import BlobFinder
import cv2
import numpy as np


class TestBlobExtractor(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestBlobExtractor, self).__init__(*args, **kwargs)
        self.run_ntimes_oneblob()

    def run_ntimes_oneblob(self):
        """
        Run the BlobFinder on one blob
        """
        self.mask, self.background = self.get_maskbackground()
        blob = self.get_randomblob(self.mask, self.background)[0]
        self.bfinder = self.get_blobfinder()
        # init background
        for _ in range(10):
            blob = self.get_randomblob(self.mask, self.background)[0]
            self.bfinder.run(blob)

        self.blobs_center = list()
        self.blobs_found = list()
        for frame_i in range(10):
            blob, blob_center, _, _ = \
                self.get_randomblob(self.mask, self.background)
            self.bfinder.run(blob)
            self.blobs_center.append(blob_center)
            self.blobs_found.append(self.bfinder.filtered_contours)

        self.blob = blob

    def get_maskbackground(self):
        background = (np.random.rand(1080, 1980) * 100).astype(np.uint8) + 128
        mask = background.copy()
        mask[:] = 0
        cv2.circle(mask, (mask.shape[1] // 2,
                          mask.shape[0] // 2),
                   mask.shape[0] // 2, (255, 0, 0), -1)
        return mask, background

    def get_randomblob(self, mask, background, color=20):
        blob = background.copy()
        blob_max_radius = mask.shape[0] // 20
        blob_min_radius = 5
        blob_center = (np.random.rand(2) - 0.5) *\
            (mask.shape[0] // 2 - blob_max_radius)
        blob_center[0] += mask.shape[1] // 2
        blob_center[1] += mask.shape[0] // 2
        blob_center = blob_center.astype(np.uint)
        blob_axes = np.random.rand(2) * (blob_max_radius - blob_min_radius)
        blob_axes += blob_min_radius
        blob_axes = blob_axes.astype(np.uint)
        blob_axes.sort()
        blob_angle = np.random.rand() * 180
        cv2.ellipse(blob,
                    (blob_center[0], blob_center[1]),
                    (blob_axes[0], blob_axes[1]),
                    blob_angle, 0, 360, color, -1)
        return blob, blob_center, blob_axes[::-1] * 2, blob_angle

    def get_blobfinder(self):
        mask, image = self.get_maskbackground()
        bfinder = BlobFinder(mask)
        bfinder.run(image)
        return bfinder

    def test_setget_image(self):
        mask, image = self.get_maskbackground()
        blob = self.get_randomblob(mask, image)[0]
        bfinder = self.get_blobfinder()
        bfinder.run(blob)
        myim = bfinder.original_image
        self.assertTrue(np.allclose(myim, blob))

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
