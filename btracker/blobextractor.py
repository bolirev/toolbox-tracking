"""
BlobExtractor is a class for extracting elliptical object from the background
"""
# import the necessary packages
import cv2
import numpy as np
import pandas as pd


class BlobFinder():

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
        if self.skip_filter_contours:
            self.__filtered_contours = None
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
        ellipse = Ellipse()
        for cont in self.__contours:
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


class BlobCroper():
    def __init__(self, xmargin=[0, 0], ymargin=[0, 0]):
        self.xmargin = xmargin
        self.ymargin = ymargin

    def crop(self, image, ellipses):
        croped_images = list()
        for ell in ellipses:
            y_spread, x_spread = ell.spread()
            y_range = np.arange(np.floor(ell.x - y_spread / 2)
                                - self.ymargin[0],
                                np.ceil(ell.x + y_spread / 2)
                                + self.ymargin[1])
            y_range = y_range.astype(np.int)
            x_range = np.arange(np.floor(ell.y - x_spread / 2)
                                - self.xmargin[0],
                                np.ceil(ell.y + x_spread / 2)
                                + self.xmargin[1])
            x_range = x_range.astype(np.int)
            y_range = np.clip(y_range, 0, image.shape[1] - 1)
            x_range = np.clip(x_range, 0, image.shape[0] - 1)
            croped_im = image[x_range, :]
            croped_im = croped_im[:, y_range]
            croped_images.append(croped_im)
        return croped_images


class BlobFilter():
    def __init__(self, histogram, bin_edges):
        self.__edges = bin_edges
        self.__refcumsum = self.hist2cumsum(histogram)

    def hist2cumsum(self, hist):
        return np.cumsum(hist / np.sum(hist))

    def image2hist(self, image):
        image = image.flatten()
        image = image[np.isnan(image) == False]
        hist, _ = np.histogram(image, bins=self.__edges)
        return hist

    def score(self, images):
        if not isinstance(images, list):
            raise TypeError('images should be a list of numpy array')
        if len(images) == 0:
            return None
        if not isinstance(images[0], np.ndarray):
            raise TypeError('images should be a list of numpy array')

        scores = np.zeros(len(images))
        for im_i in range(scores.shape[0]):
            hist = self.image2hist(images[im_i])
            cumsum = self.hist2cumsum(hist)
            score = np.sqrt(np.mean((cumsum - self.__refcumsum)**2))
            scores[im_i] = score
        return scores


class BlobMatcher():
    def __init__(self):
        self.__features = dict()
        self.__features['rdistance'] = self.rdistance
        self.__features['rarea'] = self.area
        self.__features['rangle'] = self.angle
        self.__features['rroundness'] = self.roundness
        self.weights = pd.Series(index=list(self.__features.keys()),
                                 data=0)
        self.weights.distance = 1
        self.__score_matrix = None

    @property
    def features(self):
        return list(self.__features.keys())

    def rdistance(self):
        refdim = len(self.__reference)
        tardim = len(self.__target)
        score_matrix = np.zeros(refdim, tardim)
        for refi in range(refdim):
            xref = self.__reference[refi].x
            yref = self.__reference[refi].y
            for tari in range(refi, tardim):
                xtar = self.__reference[tari].x
                ytar = self.__reference[tari].y
                score_matrix[refi, tari] = np.sqrt(
                    (xref - xtar)**2 + (yref - ytar)**2)
                score_matrix[tari, refi] = score_matrix[refi, tari]
        score_matrix = score_matrix / score_matrix.max()
        score_matrix = 1 - score_matrix
        return score_matrix

    def rroundness(self):
        refdim = len(self.__reference)
        tardim = len(self.__target)
        score_matrix = np.zeros(refdim, tardim)
        for refi in range(refdim):
            rref = self.__reference[refi].roundness
            for tari in range(refi, tardim):
                rtar = self.__reference[tari].roundness
                score_matrix[refi, tari] = np.sqrt((rref - rtar)**2)
                score_matrix[tari, refi] = score_matrix[refi, tari]
        score_matrix = score_matrix / score_matrix.max()
        score_matrix = 1 - score_matrix
        return score_matrix

    def rarea(self):
        refdim = len(self.__reference)
        tardim = len(self.__target)
        score_matrix = np.zeros(refdim, tardim)
        for refi in range(refdim):
            aref = self.__reference[refi].area
            for tari in range(refi, tardim):
                atar = self.__target[tari].area
                score_matrix[refi, tari] = np.abs(aref - atar)
                score_matrix[tari, refi] = score_matrix[refi, tari]
        score_matrix = score_matrix / score_matrix.max()
        score_matrix = 1 - score_matrix
        return score_matrix

    def rangle(self):
        refdim = len(self.__reference)
        tardim = len(self.__target)
        score_matrix = np.zeros(refdim, tardim)
        for refi in range(refdim):
            aref = self.__reference[refi].angle
            for tari in range(refi, tardim):
                atar = self.__target[tari].angle
                score_matrix[refi, tari] = np.abs(np.cos(aref - atar))
                score_matrix[tari, refi] = score_matrix[refi, tari]
        score_matrix = score_matrix / score_matrix.max()
        return score_matrix

    def __score(self):
        refdim = len(self.__reference)
        tardim = len(self.__target)
        self.__score_matrix = np.zeros(refdim, tardim)
        for key, method in self.__features:
            self.__fscore[key] = method()

        return (self.weights * self.__fscore).sum()

    def match(self, ref_ellipses, ellipses):
        self.__reference = ref_ellipses
        self.__target = ellipses
