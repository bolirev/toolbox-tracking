"""
   Reorder TimeBench data into:
   - Calibrations
   - Recordings
"""
import argparse
import os
import glob
import shutil


parser = argparse.ArgumentParser()
arghelp = 'Path to the folder containing the data'
parser.add_argument('--folder',
                    type=str,
                    default=None,
                    help=arghelp)

arghelp = 'Number of cameras'
parser.add_argument('--ncameras',
                    type=int,
                    default=3,
                    help=arghelp)


args = parser.parse_args()
print(args)
# Check that the folder exists and the subfolders as well
if not os.path.exists(args.folder):
    raise IOError('{} does not exist'.format(args.folder))

timebench_folder = 'FromTimeBench'
calibration_folder = 'Calibrations'
recordings_folder = 'Recordings'
cam_prefix = 'cam_'
extension = '.jpg'

subfolders = [timebench_folder,
              calibration_folder,
              recordings_folder]

for subf in subfolders:
    folder = os.path.join(args.folder, subf)
    if not os.path.exists(folder):
        raise IOError('{} does not exist'.format(folder))

# Get a list of all recordings in TimeBench
# recordings are a mixture of Calibrations and Flights
for folder in glob.glob(os.path.join(args.folder, timebench_folder, '*')):
    fbasename = os.path.basename(folder)
    if 'calib' in folder.lower():
        target_folder = os.path.join(args.folder,
                                     calibration_folder,
                                     fbasename)
    else:
        target_folder = os.path.join(args.folder,
                                     recordings_folder,
                                     fbasename)
    if os.path.exists(target_folder):
        continue
    os.mkdir(target_folder)
    print('copy')
    print(folder + ' to ' + target_folder)
    for cam_i in range(args.ncameras):
        tfolder_cam = os.path.join(target_folder, 'cam_{}'.format(cam_i))
        os.mkdir(tfolder_cam)
        template = os.path.join(args.folder,
                                timebench_folder,
                                '**/cam_{}*{}'.format(cam_i,
                                                      extension))
        for im_file in glob.glob(template,
                                 recursive=True):
            imbasename = os.path.basename(im_file)
            shutil.copy(im_file, os.path.join(tfolder_cam, imbasename))
