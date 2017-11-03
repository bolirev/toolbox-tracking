import unittest
import beetracktracker as bt
import cv2
import numpy as np


class TestBeeTrackTracker(unittest.TestCase):
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
        blob_angle = np.deg2rad(blob_angle)
        #blob_angle = np.arctan2(np.sin(blob_angle), np.cos(blob_angle))
        blob_angle = np.rad2deg(blob_angle)
        return blob, blob_center, blob_axes[::-1] * 2, blob_angle

    def get_mybee(self):
        mask, image = self.get_maskbackground()
        mybee = bt.beetracktracker()
        mybee.set_mask(mask)
        mybee.set_image(image)
        mybee.run()
        return mybee

    def one_blob(self):
        pass

    def test_setget_image(self):
        mask, image = self.get_maskbackground()
        blob = self.get_randomblob(mask, image)[0]
        mybee = self.get_mybee()
        mybee.set_image(blob)
        mybee.run()
        step = bt.beetracktracker.processing_steps.read
        myim = mybee.get_data(step)
        self.assertTrue(np.allclose(myim, blob))

    def test_masked(self):
        mask, image = self.get_maskbackground()
        blob = self.get_randomblob(mask, image)[0]
        mybee = self.get_mybee()
        mybee.set_image(blob)
        mybee.run()
        step = bt.beetracktracker.processing_steps.masked
        myim = mybee.get_data(step)
        testimage = cv2.bitwise_and(blob, blob, mask=mask)
        self.assertTrue(np.allclose(myim, testimage))

    def test_segmented(self):
        mask, image = self.get_maskbackground()
        blob = self.get_randomblob(mask, image)[0]
        mybee = self.get_mybee()
        mybee.set_image(blob)
        mybee.run()
        step = bt.beetracktracker.processing_steps.segmented
        myim = mybee.get_data(step)
        self.assertTrue(np.allclose(np.unique(myim), [0, 255]))

    def test_errode(self):
        mask, image = self.get_maskbackground()
        blob = self.get_randomblob(mask, image)[0]
        mybee = self.get_mybee()
        mybee.set_image(blob)
        mybee.run()
        step = bt.beetracktracker.processing_steps.segmented
        segmented = mybee.get_data(step)
        step = bt.beetracktracker.processing_steps.eroded
        eroded = mybee.get_data(step)
        _, unique_count_segmented = np.unique(
            segmented, return_counts=True)
        u, unique_count_eroded = np.unique(eroded, return_counts=True)
        print(unique_count_eroded)
        print(u)
        self.assertGreater(unique_count_eroded[0],
                           unique_count_segmented[0])


if __name__ == '__main__':
    unittest.main()
