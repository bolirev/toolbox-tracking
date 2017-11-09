"""
Different tools to filter blobs
"""
import numpy as np


def hist2cumsum(hist):
    return np.cumsum(hist / np.sum(hist))


def image2hist(image, edges):
    image = image.flatten()
    image = image[np.isnan(image) == False]
    hist, _ = np.histogram(image, bins=edges)
    return hist


def score_cumsum_hist(images, ref_cumsum, bin_edges):
    if not isinstance(images, list):
        raise TypeError('images should be a list of numpy array')
    if len(images) == 0:
        return None
    if not isinstance(images[0], np.ndarray):
        raise TypeError('images should be a list of numpy array')

    scores = np.zeros(len(images))
    for im_i in range(scores.shape[0]):
        hist = image2hist(images[im_i], bin_edges)
        cumsum = hist2cumsum(hist)
        score = np.sqrt(np.mean((cumsum - ref_cumsum)**2))
        scores[im_i] = score
    return scores
