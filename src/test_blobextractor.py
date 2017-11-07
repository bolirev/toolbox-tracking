import unittest
from blobextractor import BlobFinder
import cv2
import numpy as np


class TestBlobExtractor(unittest.TestCase):
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
        mask, image = self.get_maskbackground()
        blob = self.get_randomblob(mask, image)[0]
        bfinder = self.get_blobfinder()
        bfinder.run(blob)
        myim = bfinder.masked_image
        testimage = cv2.bitwise_and(blob, blob, mask=mask)
        self.assertTrue(np.allclose(myim, testimage))

    def test_segmented(self):
        mask, background = self.get_maskbackground()
        bfinder = self.get_blobfinder()
        for _ in range(2):
            blob = self.get_randomblob(mask, background)[0]
            bfinder.run(blob)
        myim = bfinder.segmented_image
        self.assertTrue(np.allclose(np.unique(myim), [0, 255]))

    def test_oneblob(self):
        mask, background = self.get_maskbackground()
        blob = self.get_randomblob(mask, background)[0]
        bfinder = self.get_blobfinder()
        # init background
        for _ in range(10):
            blob = self.get_randomblob(mask, background)[0]
            bfinder.run(blob)

        condition = True
        for frame_i in range(10):
            blob, blob_center, _, blob_angle = self.get_randomblob(
                mask, background)
            bfinder.run(blob)
            contours = bfinder.filtered_contours
            if len(contours) > 1:
                print('Too many blobs detected')
                condition = False
                continue
            elif len(contours) < 1:
                print('Too few blobs detected')
                condition = False
                continue
            contours = contours[0]
            px_error = np.sqrt(np.sum(contours[0] - blob_center)**2)
            if px_error > 1:
                print('Pixel error too large : {}'.format(px_error))
                condition = False
        #
        self.assertTrue(condition)


if __name__ == '__main__':
    unittest.main()
