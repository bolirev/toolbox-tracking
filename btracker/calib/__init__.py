# Python init file
import pandas as pd
import numpy as np
import cv2
from mcameras.io import ivfile


def concatenate_manhattan_2d3d(manhattan_3d, manhattan_2d):
    if not isinstance(manhattan_2d, pd.DataFrame):
        raise TypeError('manhattan_2d should be a DataFrame')
    if not isinstance(manhattan_3d, pd.DataFrame):
        raise TypeError('manhattan_3d should be a DataFrame')
    d = {'three_d': manhattan_3d, 'two_d': manhattan_2d}
    manhattan_3d2d = pd.concat(d.values(), axis=1, keys=d.keys())
    manhattan_3d2d = manhattan_3d2d.astype(float)
    return manhattan_3d2d


def objects_points_from_manhattan(manhattan_3d2d):
    if not isinstance(manhattan_3d2d, pd.DataFrame):
        raise TypeError('manhattan_3d2d should be a DataFrame')
    return manhattan_3d2d.dropna().loc[:, 'three_d']\
                                  .loc[:, ['x', 'y', 'z']].values


def image_points_from_manhattan(manhattan_3d2d):
    if not isinstance(manhattan_3d2d, pd.DataFrame):
        raise TypeError('manhattan_3d2d should be a DataFrame')
    return manhattan_3d2d.dropna().loc[:, 'two_d'].loc[:, ['x', 'y']].values


def campose_from_vectors(rvec, tvec):
    r = cv2.Rodrigues(rvec)[0]
    camera_pose = np.hstack((r, tvec))
    camera_pose = np.vstack((camera_pose, [0, 0, 0, 1]))
    return camera_pose


def camera_pose(manhattan_2d, manhattan_3d, camera_intrinsics):
    camera_matrix = camera_intrinsics['intrinsic_matrix']
    distcoeffs = camera_intrinsics['distortion']

    manhattan_2d3d = concatenate_manhattan_2d3d(manhattan_3d, manhattan_2d)
    objects_points = objects_points_from_manhattan(manhattan_2d3d)
    image_points = image_points_from_manhattan(manhattan_2d3d)

    retval, rvec, tvec, inliers = cv2.solvePnPRansac(
        objects_points, image_points, camera_matrix, distcoeffs)

    return campose_from_vectors(rvec, tvec)


def calibrates_extrinsic(filenames,
                         file_manhattan_3d,
                         cameras_intrinsics,
                         corner_th=-1,
                         manhattan_3d_key='manhattan'):

    cameras_extrinsics = dict()
    manhattan_3d = pd.read_hdf(file_manhattan_3d, manhattan_3d_key)
    for cam_i, cfile in filenames.items():
        manhattan_2d = ivfile.manhattan(cfile, corner_th)
        cameras_extrinsics[cam_i] = dict()
        cameras_extrinsics[cam_i]['pose'] = camera_pose(
            manhattan_2d, manhattan_3d, cameras_intrinsics[cam_i])
    return cameras_extrinsics
