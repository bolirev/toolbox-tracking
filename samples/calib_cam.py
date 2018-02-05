"""
   Create folder containing:

"""


import glob
import os
import numpy as np
import pandas as pd
import xml.etree.cElementTree as ET

calib_folder = '/home/bolirev/Desktop/TestCalib/180110_114708_calib'


def indent(elem, level=0):
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def test_xml_imagelist_calib(xml_file):
    tree = ET.ElementTree(file=xml_file)
    roottag = 'opencv_storage'
    root = tree.getroot()
    if root.tag != roottag:
        raise IOError('{} does not contain {}.'.format(xml_file, roottag))
    foundimage_list = False
    for child_of_root in root:
        tag = 'images'
        if child_of_root.tag != tag:
            continue
        if foundimage_list:
            raise KeyError(
                '{} contains two or more image list (tag P{})'.format(xml_file,
                                                                      tag))
        foundimage_list = True
        images = pd.DataFrame(
            columns=['path', 'cam_idx', 'time_stamp', 'status', 'exist'])
        image_pattern = None
        image_i = 0
        for line in child_of_root.text.splitlines():
            line = line.strip()
            if not line:
                continue  # empty line
            if image_pattern is None:
                image_pattern = line
            else:
                images.loc[image_i, 'path'] = line
                imname = os.path.splitext(os.path.basename(line))[0]
                iminfo = imname.split('-')
                if len(iminfo) == 2:
                    cam_idx = iminfo[0]
                    time_stamp = float(iminfo[1])
                    status = 'Ok'
                else:
                    cam_idx = None
                    time_stamp = None
                    status = 'ERROR Parsing:{}'.format(iminfo)
                exist = os.path.exists(line)
                images.loc[image_i, 'path'] = line
                images.loc[image_i, 'status'] = status
                images.loc[image_i, 'cam_idx'] = cam_idx
                images.loc[image_i, 'time_stamp'] = time_stamp
                images.loc[image_i, 'exist'] = exist
            image_i += 1
    # Now we can test if files is correctly formated
    raiseerror = False
    if not os.path.exists(image_pattern):
        print('{} can not be found'.format(image_pattern))
        raiseerror = True
    if not np.all(images.exist):
        print(images.loc[not images.exist, :])
        raiseerror = True
    if not np.all(images.status == 'Ok'):
        print(images.loc[images.exist != 'Ok', :])
        raiseerror = True
    # Check if every camera time stamp
    listofcam = np.unique(images.loc[:, 'cam_idx'].dropna())
    listoftime_stamp = np.unique(images.loc[:, 'time_stamp'].dropna()).tolist()
    for cam_i in listofcam:
        tstamps = images.time_stamp.loc[images.cam_idx == cam_i].tolist()
        for ts in listoftime_stamp:
            if ts not in tstamps:
                print('Cam {} miss time stamp {}'.format(cam_i, ts))
                raiseerror = True
    if raiseerror:
        raise IOError('Parsing error')
    return True


def create_imagelist_calib(imgfolder, calib_list='calib_list.xml'):
    """
    """
    calib_list = os.path.join(imgfolder, calib_list)
    filelist = list()
    fname = os.path.join(imgfolder, 'pattern.png')
    absfname = os.path.abspath(fname)
    filelist.append(absfname)
    for fname in glob.iglob(imgfolder + '/**/*.jpg', recursive=True):
        absfname = os.path.abspath(fname)
        if absfname == filelist[0]:
            continue
        filelist.append(absfname)
    filelist_xml = ET.Element('images')
    filelist_xml.text = """{}""".format("\n".join(filelist))
    root = ET.Element('opencv_storage')
    root.append(filelist_xml)
    tree = ET.ElementTree(root)
    indent(root)
    tree.write(calib_list)
    return calib_list


calib_list = create_imagelist_calib(calib_folder)
test_xml_imagelist_calib(calib_list)
