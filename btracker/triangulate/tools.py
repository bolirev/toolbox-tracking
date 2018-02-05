import numpy as np
import cv2
from scipy.sparse.linalg import svds
from functools import partial


def emsvd(Y, k=None, tol=1E-3, maxiter=None):
    """
    Approximate SVD on data with missing values via expectation-maximization

    :param Y: (nobs, ndim) data matrix, missing values denoted by NaN/Inf
    :param k: number of singular values/vectors to find (default: k=ndim)
    :param tol: convergence tolerance on change in trace norm
    :param maxiter: maximum number of EM steps to perform (default: no limit)
    :returns: Y_hat, mu_hat, U, s, Vt
    Y_hat:      (nobs, ndim) reconstructed data matrix
    mu_hat:     (ndim,) estimated column means for reconstructed data
    U, s, Vt:   singular values and vectors (see np.linalg.svd and
                scipy.sparse.linalg.svds for details)

    Methods for large scale SVD with missing values
    Miklós Kurucz, András A. Benczúr, Károly Csalogány, 2007
    """

    if k is None:
        svdmethod = partial(np.linalg.svd, full_matrices=False)
    else:
        svdmethod = partial(svds, k=k)
    if maxiter is None:
        maxiter = np.inf

    # initialize the missing values to their respective column means
    mu_hat = np.nanmean(Y, axis=0, keepdims=1)
    valid = np.isfinite(Y)
    Y_hat = np.where(valid, Y, mu_hat)

    halt = False
    ii = 1
    v_prev = 0

    while not halt:

        # SVD on filled-in data
        U, s, Vt = svdmethod(Y_hat - mu_hat)

        # impute missing values
        Y_hat[~valid] = (U.dot(np.diag(s)).dot(Vt) + mu_hat)[~valid]

        # update bias parameter
        mu_hat = Y_hat.mean(axis=0, keepdims=1)

        # test convergence using relative change in trace norm
        v = s.sum()
        if ii >= maxiter or ((v - v_prev) / v_prev) < tol:
            halt = True
        ii += 1
        v_prev = v

    return Y_hat, mu_hat, U, s, Vt


def random_projects_points(npoints, edge_length, cameras_calib):
    ncameras = len(cameras_calib)
    pts_cam = list()
    pts_3d = (np.random.rand(npoints, 3) - 0.5) * 2
    pts_3d[:, 2] += 1
    pts_3d[:, 2] *= edge_length
    pts_3d[:, 1] *= edge_length
    pts_3d[:, 0] *= edge_length

    for cam_i in range(ncameras):
        cam_pose = cameras_calib[cam_i]['pose']
        cam_mat = cameras_calib[cam_i]['intrinsic_matrix']
        cam_dist = cameras_calib[cam_i]['distortion']
        rvec, jacobian = cv2.Rodrigues(cam_pose[:3, :3])
        tvec = cam_pose[:3, 3]
        impoints, jacobian = cv2.projectPoints(
            pts_3d, rvec, tvec, cam_mat, cam_dist)
        pts_cam.append(np.squeeze(impoints))
    return pts_cam, pts_3d


def undistord_ncam_points(cameras_calib, pts_cam):
    ncameras = len(cameras_calib)
    if np.shape(pts_cam)[0] != ncameras:
        raise IndexError(
            'Cameras points should have the same length than cameras_calib')
    if len(pts_cam.shape) == 2:
        pts_cam = pts_cam[:, np.newaxis, :]
    if len(pts_cam.shape) != 3:
        raise IndexError('Camera points should be of shape 3')
    for cam_i in range(ncameras):
        dst = cv2.undistortPoints(pts_cam[cam_i, ...][np.newaxis, ...],
                                  cameras_calib[cam_i]['intrinsic_matrix'],
                                  cameras_calib[cam_i]['distortion'])
        pts_cam[cam_i, ...] = np.squeeze(dst)

    return pts_cam
