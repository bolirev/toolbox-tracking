"""
  A window
"""
import cv2
import numpy as np
from btracker.blobs.finder import BlobFinder
import btracker.tools.generator as btgen
from btracker.plot.image import overlay_ellipses
import yaml
import glob
import argparse
import btracker.io.ivfile as btio
import os
import pandas as pd
import time
from btracker.blobs.croped import crop
from scipy import signal

def load_config(config_yaml):
    with open(config_yaml, 'r') as stream:
        try:
            config = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return config

config_yaml =  'cropper.yaml'

rootdir = config['files']['rootdir']
imagdir = config['files']['images']
tracdir = config['files']['tracking']
# Create template_frame
template_frame = os.path.join(rootdir,imagdir)
tbee = '*bee_{'+config['bee']['format']+'}*'
template_frame = os.path.join(template_frame,tbee.format(config['bee']['number']))

template_tra = os.path.join(rootdir, tracdir)
tbee = 'bee_{'+config['bee']['format']+'}'
template_tra = os.path.join(template_tra,tbee.format(config['bee']['number']))

# Parameter for croping
xmargin = config['cropper']['xmargin']
ymargin = config['cropper']['ymargin']

for cam_i in range(config['camera']['ncamera']):
    camtemplate_frame = os.path.join(template_frame,'cam_%d'.format(cam_i))
    camtemplate_frame = os.path.join(camtemplate_frame,'*.jpg')
    print(camtemplate_frame)
    tra_file = os.path.join(template_tra,'cam_%d.tra'.format(cam_i))
    print(tra_file)

    #Load image filenames
    filelist = sorted(glob.glob(template_frame))
    maxframe = len(filelist)

    #Load trafile
    tra_data=btio.load(tra_file)

    frame_i = 0
    folder_tosave_cropped = os.path.dirname(filelist[frame_i])
    while True:
        # Check that frame number is not above last one
        # and previous one
        if frame_i >= (maxframe):
            break
        # grab the current frame and initialize the occupied/unoccupied
        imname = filelist[frame_i]
        frame = cv2.imread(imname, 0)  # 0 for blackwhite

        # Load ellipse, and set as a point since we want
        # croping intependent of ellipse size
        curr_ell=tra_data.loc[frame_i,0]
        curr_ell.roundness=1
        curr_ell.size=0
        curr_ell.angle=0

        # Crop image
        croped_im = crop(frame, [curr_ell], xmargin, ymargin)[0]

        # Save cropped image
        imagename = os.path.join('bee_'+folder_tosave_cropped, os.path.basename(imname))
        cv2.imwrite(imagename,croped_im)
