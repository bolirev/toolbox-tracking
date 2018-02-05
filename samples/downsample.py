import shutil
import glob
import os
import argparse


parser = argparse.ArgumentParser()
arghelp = 'The path to the folder containing the images to be downsampled'
parser.add_argument('--folder',
                    type=str,
                    default=None,
                    help=arghelp)
arghelp = 'The downsampling factor, i.e. save every n images with n ' \
    + 'being the downsampling factor.'
parser.add_argument('--nskip',
                    type=int,
                    default=None,
                    help=arghelp)
arghelp = 'The extension of the images (default .jpg)'
parser.add_argument('--extension',
                    type=str,
                    default='.jpg',
                    help=arghelp)


args = parser.parse_args()
print(args)
if not os.path.exists(args.folder):
    raise IOError('{} does not exist'.format(args.folder))
if not (args.nskip):
    raise IOError('{} should be a positive integer'.format(args.nskip))


def skipevery_nimage(folder, downsamp_factor, extension):
    folder_downsamp = folder + '_downsamp'
    skip = 0
    if os.path.exists(folder_downsamp):
        shutil.rmtree(folder_downsamp)
    os.mkdir(folder_downsamp)
    for cfile in sorted(glob.glob(os.path.join(folder,
                                               '*' + extension))):
        if skip == 0:
            cbasename = os.path.basename(cfile)
            to_file = os.path.join(folder_downsamp, cbasename)
            shutil.copy(cfile, to_file)
            skip = downsamp_factor - 1
        else:
            skip -= 1


skipevery_nimage(folder=args.folder,
                 downsamp_factor=args.nskip,
                 extension=args.extension)
