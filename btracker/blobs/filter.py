"""
Different tools to filter blobs
"""
import numpy as np


def hist2cumsum(self, hist):
    return np.cumsum(hist / np.sum(hist))


def image2hist(self, image):
    image = image.flatten()
    image = image[np.isnan(image) == False]
    hist, _ = np.histogram(image, bins=self.__edges)
    return hist


def score_cumsum_hist(self, images, ref_cumsum, bin_edges):
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
