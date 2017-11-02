import unittest
import beetracktracker as bt
import cv2
import numpy as np


class TestBeeTrackTracker(unittest.TestCase):
    def get_testimages(self):
        mask = cv2.imread('data/BackWhite_mask.jpg',
                          cv2.IMREAD_GRAYSCALE)
        image = cv2.imread('data/frame_00000000.jpg')
        return mask, image

    def get_mybee(self):
        mask, image = self.get_testimages()
        mybee = bt.beetracktracker()
        mybee.set_mask(mask)
        mybee.set_image(image)
        return mybee

    def test_setget_image(self):
        mask, image = self.get_testimages()
        mybee = self.get_mybee()
        step = bt.beetracktracker.processing_steps.read
        myim = mybee.get_data(step)
        self.assertTrue(np.allclose(myim, image))

    def test_masked(self):
        mask, image = self.get_testimages()
        mybee = self.get_mybee()
        step = bt.beetracktracker.processing_steps.masked
        myim = mybee.get_data(step)
        testimage = cv2.bitwise_and(image, image, mask=mask)

        self.assertTrue(np.allclose(myim, testimage))


if __name__ == '__main__':
    unittest.main()
