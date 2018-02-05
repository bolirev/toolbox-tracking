import cv2
import os
import numpy as np
import xml.etree.ElementTree as ET


def load(filename, key):
    cv_file = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
    matrix = cv_file.getNode(key).mat()
    cv_file.release()
    return matrix


def save(filename, key, data, overwrite=False):
    if os.path.exists(filename) and (overwrite is False):
        cv_file = cv2.FileStorage(filename, cv2.FILE_STORAGE_APPEND)
    else:
        cv_file = cv2.FileStorage(filename, cv2.FILE_STORAGE_WRITE)
    cv_file.write(key, data)
    cv_file.release()


def intrinsic_matrix(filename, key="intrinsic_matrix"):
    return load(filename, key)


def distortion(filename, key="distortion"):
    return load(filename, key)


def cameras_intrinsic_calibration(filenames):
    """Load the intrinsic calibration from files

    :param filenames:dictionary of filenames, here the key is the camera name
    :returns: a dictionary containing instrinsic_matrix and distortions
    :rtype: dict

    """

    cameras_intrinsics = dict()
    for cam_i, cfile in filenames.items():
        cameras_intrinsics[cam_i] = dict()
        cameras_intrinsics[cam_i]['intrinsic_matrix'] = intrinsic_matrix(cfile)
        cameras_intrinsics[cam_i]['distortion'] = distortion(cfile)
    return cameras_intrinsics


def load_cameras_calibration(filename):
    """Load the intrinsic and intrinsic calibration from file

    :param filenames:calibration filename
    :returns: a dictionary containing intrinsic and extrinsic calib
    :rtype: dict

    """
    tree = ET.parse(filename)
    root = tree.getroot()

    ncameras_key = 'ncameras'
    ncameras = 0
    for ncameras_node in root.iter(ncameras_key):
        ncameras = np.round(float(ncameras_node.text)).astype(int)
        break
    if ncameras == 0:
        raise ValueError('ncameras is missing in xml file')

    cameras_calib = dict()
    for cam_i in range(ncameras):
        cameras_calib[cam_i] = dict()
        for key in ['intrinsic_matrix', 'pose', 'distortion']:
            cameras_calib[cam_i][key] = load(filename,
                                             key + '_{}'.format(cam_i))
    return cameras_calib


def save_cameras_calibration(cameras_intrinsics, cameras_extrinsics,
                             filename, overwrite=False):
    """Save the intrinsic and intrinsic calibration
    """
    ncameras_key = 'ncameras'
    ncameras = len(cameras_intrinsics)
    if len(cameras_extrinsics) != ncameras:
        raise KeyError(
            'Number of cameras differ between intrinsics, and extrinsics')

    save(filename, ncameras_key, ncameras, overwrite)

    for cam_i in range(ncameras):
        for key in ['pose']:
            data = cameras_extrinsics[cam_i][key]
            save(filename, key + '_{}'.format(cam_i), data)
        for key in ['intrinsic_matrix', 'distortion']:
            data = cameras_intrinsics[cam_i][key]
            save(filename, key + '_{}'.format(cam_i), data)
