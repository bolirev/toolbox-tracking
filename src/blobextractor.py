"""
BlobExtractor is a class for extracting elliptical object from the background
"""
# import the necessary packages
import cv2
import numpy as np
print(cv2.__version__)


class BlobExtractor():

    def __init__(self, mask_image):
        # Define get only variable variable
        self.__orig_image = None
        self.__masked_image = None
        self.__segmented_image = None
        self.__blured_image = None
        self.__eroded_image = None
        self.__dilated_image = None
        self.__processed_image = None
        self.__contours = None
        self.__filtered_contours = None
        self.__cropped_images = None
        # Define 'private' variable
        self.__fgbg = cv2.createBackgroundSubtractorKNN()
        # Declare parameters
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
        self.skip_crop = False

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
        self.__filtered_contours = None
        return toreturn

    @property
    def croped_images(self):
        if self.skip_crop:
            self.crop()
        toreturn = self.__cropped_images
        self.__cropped_images = None
        return toreturn

    def run(self, image):
        if not isinstance(image, np.ndarray):
            raise TypeError('mask_image is not an ndarray')
        if image.ndim != self.mask_image.ndim:
            raise TypeError('mask and image should have the same dimension')
        if not np.allclose(image.shape, self.mask_image.shape):
            raise TypeError('mask and image should have the same dimension')
        self.__orig_image = image
        self.mask()
        self.blur()
        self.segment()
        self.dilate()
        self.erode()
        self.__processed_image = self.__eroded_image
        self.find_contours()
        if not self.skip_filter_contours:
            self.filter_contours()
            if not self.skip_crop:
                self.crop()

    def mask(self):
        if self.mask_image is not None:
            self.__masked_image = cv2.bitwise_and(self.__orig_image,
                                                  self.__orig_image,
                                                  mask=self.mask_image)
        else:
            self.__masked_image = self.__orig_image

    def blur(self):
        if self.gaussian_blur < 2:
            self.__blured_image = self.__masked_image
        else:
            self.__blured_image = cv2.GaussianBlur(self.__masked_image,
                                                   (self.gaussian_blur,
                                                    self.gaussian_blur), 0)

    def segment(self):
        self.__segmented_image = self.__fgbg.apply(self.__blured_image)
        self.__segmented_image = cv2.threshold(self.__segmented_image,
                                               self.threshold,
                                               255,
                                               cv2.THRESH_BINARY)[1]

    def dilate(self):
        self.__dilated_image = cv2.dilate(self.__segmented_image, None,
                                          iterations=self.dilate_iter)

    def erode(self):
        self.__eroded_image = cv2.erode(self.__dilated_image, None,
                                        iterations=self.erode_iter)

    def find_contours(self):
        _, self.__contours, _ = cv2.findContours(self.__processed_image,
                                                 cv2.RETR_TREE,
                                                 cv2.CHAIN_APPROX_SIMPLE)

    def filter_contours(self):
        # loop over the contours
        self.__filtered_contours = list()
        for cont in self.__contours:
            # The blob should be fitted with an ellipse
            # We can not do that if the contours contain less than 5 points
            if len(cont) < 5:
                continue
            # if the contour is too small, ignore it
            if (cv2.contourArea(cont) < min(self.area_lim)) or \
                    (cv2.contourArea(cont) > max(self.area_lim)):
                continue
            ellipse = cv2.fitEllipse(cont)
            # Roundness
            roundness = ellipse[1][0] / ellipse[1][1]
            if (roundness < min(self.roundness_lim)) or \
               (roundness > max(self.roundness_lim)):
                continue
            # Everything passed
            self.__filtered_contours.append(ellipse)

    def crop(self):
        if self.skip_filter_contours:
            self.filter_contours()
        contours = self.__filtered_contours
        if self.skip_filter_contours:
            self.__filtered_contours = None
        for cont in contours:
            pass
