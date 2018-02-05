import os
import glob
from multiprocessing import Pool

rootfolder = '/media/bolirev/Bombus_ternarius/Bumblebee_homing_video_3bars'

shared_param=dict()
shared_param['side']={
	'--mask':'"/media/bolirev/Bombus_ternarius/param_tracking/mask_side.jpg"',
	'--bin-edges-file':'"/media/bolirev/Bombus_ternarius/param_tracking/side_binedges.npy"',
	'--refcumsum-file':'"/media/bolirev/Bombus_ternarius/param_tracking/side_refcumsum.npy"',
	'--max-nbbee': '4', '--min-area': '40', '--max-area': '1200'}

shared_param['top']={
	'--mask':'"/media/bolirev/Bombus_ternarius/param_tracking/mask_top.jpg"',
	'--bin-edges-file':'"/media/bolirev/Bombus_ternarius/param_tracking/top_binedges.npy"',
	'--refcumsum-file':'"/media/bolirev/Bombus_ternarius/param_tracking/top_refcumsum.npy"',
	'--max-nbbee': '4', '--min-area': '40', '--max-area': '1200'}		

def process(param):
        command="python simple_tracker_v2.py "
        for key,val in param.items():
                command=command+key+' '+val+' '
        folder_path = os.path.dirname(param["--folder"])
        filenametra = os.path.join(folder_path, 'trajectory_v2.tra')
        if os.path.exists(filenametra):
                return 'TRAEXIST:'+command
        try:
                os.system(command)
        except:
                return 'ERROR:'+command
        return command

if __name__ == '__main__':
        param_list = [];
        print('Building the list files to be tracked')
        for cam in ['top','side']:
                print(os.path.join(rootfolder,'**/downsamp_'+cam+'/'+cam+'/'))
                for fname in glob.iglob(os.path.join(rootfolder,'**/downsamp_'+cam+'/'),recursive=True):
                        param_list.append(shared_param[cam].copy())
                        param_list[-1]['--folder']=os.path.join(fname,cam+'_%05d.jpg')
        print('{} will be processed'.format(len(param_list)))
        p = Pool(7)
        print(p.map(process, param_list))
