"""
   Find and filter ellipsoidal blob in an image
"""
# import the necessary packages
import cv2
import numpy as np
from btracker.tools.geometry import Ellipse


class BlobFinder():

    def __init__(self, mask_image, fgbg=cv2.createBackgroundSubtractorKNN()):
        # Define get only variable variable
        self.__orig_image = None
        self.__masked_image = None
        self.__segmented_image = None
        self.__thresholded_image = None
        self.__blured_image = None
        self.__eroded_image = None
        self.__dilated_image = None
        self.__processed_image = None
        self.__contours = None
        self.__filtered_contours = None
        # Define 'private' variable
        self.__fgbg = fgbg
        # Declare parameters
        if mask_image is not None:
            if not isinstance(mask_image, np.ndarray):
                raise TypeError('mask_image is not an ndarray')
            if mask_image.ndim != 2:
                raise TypeError('mask_image should have 2 dimension')
        self.mask_image = mask_image
        self.erode_iter = 2
        self.dilate_iter = 2
        self.area_lim = [0, 100000]
        self.roundness_lim = [0, 1]
        self.background_init = 60
        self.threshold = 10
        self.gaussian_blur = 21
        self.skip_filter_contours = False

    @property
    def original_image(self):
        return self.__orig_image

    @property
    def masked_image(self):
        return self.__masked_image

    @property
    def blured_image(self):
        return self.__blured_image

    @property
    def segmented_image(self):
        return self.__segmented_image

    @property
    def thresholded_image(self):
        return self.__thresholded_image

    @property
    def dilated_image(self):
        return self.__dilated_image

    @property
    def eroded_image(self):
        return self.__eroded_image

    @property
    def processed_image(self):
        return self.__processed_image

    @property
    def contours(self):
        return self.__contours

    @property
    def filtered_contours(self):
        if self.skip_filter_contours:
            self.filter_contours()
        toreturn = self.__filtered_contours
        if self.skip_filter_contours:
            self.__filtered_contours = None
        return toreturn

    def run(self, image):
        if not isinstance(image, np.ndarray):
            raise TypeError('image is not an ndarray')
        if self.mask_image is not None:
            if image.ndim != self.mask_image.ndim:
                raise TypeError(
                    'mask and image should have the same dimension')
            if not np.allclose(image.shape, self.mask_image.shape):
                raise TypeError(
                    'mask and image should have the same dimension')
        self.__orig_image = image
        self.__last_step = image
        self.mask()
        self.blur()
        self.segment()
        self.binarize()
        self.erode()
        self.dilate()
        self.__processed_image = self.__last_step
        self.find_contours()
        if not self.skip_filter_contours:
            self.filter_contours()

    def mask(self):
        if self.mask_image is not None:
            self.__last_step = cv2.bitwise_and(self.__last_step,
                                                  self.__last_step,
                                                  mask=self.mask_image)
        self.__masked_image = self.__last_step.copy()
        
    def blur(self):
        if self.gaussian_blur > 2:
            self.__last_step = cv2.GaussianBlur(self.__last_step,
                                                   (self.gaussian_blur,
                                                    self.gaussian_blur), 0)
        self.__blured_image = self.__last_step.copy()

    def segment(self):
        if self.__fgbg is not None:
            self.__last_step = self.__fgbg.apply(self.__last_step)
        self.__segmented_image = self.__last_step.copy()
            
    def binarize(self):
        self.__last_step = cv2.threshold(self.__last_step,
                                               self.threshold,
                                               255,
                                               cv2.THRESH_BINARY)[1]
        self.__thresholded_image = self.__last_step.copy()
        
    def dilate(self):
        self.__last_step = cv2.dilate(self.__last_step, None,
                                          iterations=self.dilate_iter)
        self.__dilated_image = self.__last_step.copy()

    def erode(self):
        self.__last_step = cv2.erode(self.__last_step, None,
                                        iterations=self.erode_iter)
        self.__eroded_image = self.__last_step.copy()

    def find_contours(self):
        _, self.__contours, _ = cv2.findContours(self.__processed_image,
                                                 cv2.RETR_TREE,
                                                 cv2.CHAIN_APPROX_SIMPLE)

    def filter_contours(self):
        # loop over the contours
        self.__filtered_contours = list()
        for cont in self.__contours:
            ellipse = Ellipse()
            # The blob should be fitted with an ellipse
            # We can not do that if the contours contain less than 5 points
            if len(cont) < 5:
                continue
            # if the contour is too small, ignore it
            if (cv2.contourArea(cont) < min(self.area_lim)) or \
                    (cv2.contourArea(cont) > max(self.area_lim)):
                continue
            ellipse.from_opencv_tuples(cv2.fitEllipse(cont))
            # Roundness
            roundness = ellipse.roundness
            if (roundness < min(self.roundness_lim)) or \
               (roundness > max(self.roundness_lim)):
                continue
            # Everything passed
            self.__filtered_contours.append(ellipse)
