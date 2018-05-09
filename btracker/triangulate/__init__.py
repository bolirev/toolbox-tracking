import cv2
import numpy as np
import pandas as pd
from btracker.triangulate.tools import undistord_ncam_points, \
    random_projects_points, emsvd


def triangulate_pair(cameras_calib, pts_cam, cam_indeces):
    """
    Triangulate two points on two cameras 
    """
    cam_i = cam_indeces[0]
    cam_j = cam_indeces[1]
    point_4d_hom = cv2.triangulatePoints(cameras_calib[cam_i]['pose'][:3],
                                         cameras_calib[cam_j]['pose'][:3],
                                         pts_cam[cam_i][:, np.newaxis, :],
                                         pts_cam[cam_j][:, np.newaxis, :])

    point_4d = point_4d_hom / np.tile(point_4d_hom[-1, :], (4, 1))
    return point_4d[:3, :].T


def triangulate_multiview_single_pts(cameras_calib, pts_cam):
    """
    Methods for large scale SVD with missing values
    Miklós Kurucz, András A. Benczúr, Károly Csalogány, 2007
    """
    ncameras = len(cameras_calib)
    if len(pts_cam) != ncameras:
        raise IndexError(
            'Cameras points should have the same length than cameras_calib')
    # Undistord the camera point via the camera model
    pts_cam = np.squeeze(undistord_ncam_points(cameras_calib, pts_cam))
    #
    A = np.zeros((4, ncameras * 2))

    for cam_i in range(ncameras):
        pose = cameras_calib[cam_i]['pose'][:3]
        idx = 2 * cam_i
        A[:, idx: (idx + 2)] = \
            pts_cam[cam_i, :][np.newaxis, :] * pose[2, :][:, np.newaxis] -\
            pose[0:2].transpose()

    _, _, _, _, values = emsvd(A.transpose())
    X = values[-1, :]
    X = X / X[-1]
    point3d = X[:3]
    return point3d


def triangulate_multiview(cameras_calib, pts_cam):
    npoints = np.shape(pts_cam)[1]
    point3d = np.zeros((npoints, 3))
    for p_i in range(npoints):
        point3d[p_i, :] = triangulate_multiview_single_pts(
            cameras_calib, pts_cam[:, p_i, :])
    return point3d


def triangulate_ncam_pairwise(cameras_calib, pts_cam):
    ncameras = len(cameras_calib)
    if pts_cam.shape[0] != ncameras:
        raise IndexError(
            'Cameras points should have the same length than cameras_calib')
    # Undistord the camera point via the camera model
    pts_cam = undistord_ncam_points(cameras_calib, pts_cam)

    # Init variables
    n_points = pts_cam.shape[1]
    max_comb = int(ncameras * (ncameras - 1) / 2)
    point_3d = np.nan * np.zeros((max_comb, 3, n_points))
    nvalid_comb = np.zeros((1, n_points))
    comb_i = 0

    # Reconstruct pairwise (every combination)
    for cam_i in range(ncameras):
        for cam_j in range(cam_i + 1, ncameras):
            cpoint_3d = triangulate_pair(
                cameras_calib, pts_cam, [cam_i, cam_j])

            cvalid_id = np.any(np.isnan(cpoint_3d) != 1, axis=1)
            nvalid_comb[:, cvalid_id] += 1
            point_3d[comb_i, :, cvalid_id] = cpoint_3d[cvalid_id, :]
            comb_i += 1

    return (np.nansum(point_3d, axis=0) / np.tile(nvalid_comb, (3, 1))).T,point_3d,nvalid_comb


def error_reconstruction(npoints, edge_length, cameras_calib):
    pts_cam, pts_3d = random_projects_points(
        npoints, edge_length, cameras_calib)
    point_3d = triangulate_ncam_pairwise(cameras_calib, pts_cam)

    point_3d = pd.DataFrame(data=point_3d, columns=['x', 'y', 'z'])
    pts_3d = pd.DataFrame(data=pts_3d, columns=['x', 'y', 'z'])
    error = point_3d - pts_3d
    d = {'reference': pts_3d, 'reconstructed': point_3d, 'error': error}
    error_df = pd.concat(d.values(), axis=1, keys=d.keys())
    error_df[('error', 'euclidian')] = np.sqrt(
        error_df.loc[:, ('error', 'x')]**2
        + error_df.loc[:,
                       ('error', 'y')]**2
        + error_df.loc[:, ('error', 'z')]**2)
    error_df[('error', 'log10_euclidian')] = np.log10(
        error_df[('error', 'euclidian')])
    return error_df
