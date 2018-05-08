import os
import glob
from multiprocessing import Pool

rootfolder = '/media/bolirev/3F7A56957DF052FD/data/20180114_6MarkersBumblebeesLearningFlights/Recordings/'

def process(param):
        command="python background_remover.py "
        for key,val in param.items():
                command=command+key+' '+val+' '
        folder_path = os.path.dirname(param["--folder"])
        filenametra = os.path.join(folder_path, 'trajectory.tra')
        #if os.path.exists(filenametra):
        #        return 'TRAEXIST:'+command
        try:
                os.system(command)
        except:
                return 'ERROR:'+command
        return command

if __name__ == '__main__':
        param_list = [];
        print('Building the list files to be tracked')
        for fname in glob.iglob(os.path.join(rootfolder,
                                             '**/cam*/**/cam*/**/cam*/'),recursive=True):
                cam=os.path.basename(fname[:-1])
                print(fname,cam)
                params = dict()
                param_list.append(params)
                param_list[-1]['--folder']=os.path.join(fname,
                                                        cam+'_%08d.jpg')
        print('{} will be processed'.format(len(param_list)))
        p = Pool(8)
        print(p.map(process, param_list))
