import pandas as pd
import numpy as np
def load_tra_file(filename_tra,separator=' '):
    """
        Read a trajectory file created by ivTrace
        The separator in the file is repeated such, 
        that columns are alligned in a text editor.
        
        :param filename_tra: filename of the trajectory to load
        :type filename_tra: string
        :param separator:(optional, default a space), the separator in the tra file
        :type separator: string
        :return: a pandas dataframe
    """
    name_param_ellipse=get_ellipse_param()
    nbcolumn=-np.inf
    #Count the number of separators    
    with open(filename_tra, 'r') as f:
        for line_i,line in enumerate(f):
            nbcol=len(line.strip().split(separator))
            if nbcol>nbcolumn:
                nbcolumn=nbcol
                #print(line_i,nbcolumn)
    nb_line=line_i
    #print(nb_line)
    
    #Create a dataframe with the number of detected columns and lines
    data=pd.DataFrame(index=np.arange(nb_line),columns=np.arange(nbcol))
    #Populate the dataframe with the data
    #Empty columns are skipped in the following manner:
    #  if 1st column is empty in the file, and the 2nd is not,
    #the 2nd column will be at the first column of data
    import re
    nums = re.compile(r"[+-]?\d+(?:\.\d+)?")
    with open(filename_tra, 'r') as f:
        for line_i,line in enumerate(f):
            #print(line_i)
            m = nums.search(line)
            frame_i=int(m.group(0)) #'{:7d}'
            line=line[(line.find(m.group(0))+len(m.group(0))):]
            data.loc[line_i,0]=frame_i
            col_i=1
            while len(line)>39:
            	cline=line[:8]
            	if cline=='        ':
            		field=np.nan
            	else:
            		field=float(cline) #' {:7.2f}'
            	line=line[8:] #Truncate line it is easier to process later
            	data.loc[line_i,col_i]=field
            	col_i+=1
            	cline=line[:8]
            	if cline=='        ':
            		field=np.nan
            	else:
            		field=float(cline) #' {:7.2f}'
            	line=line[8:] #Truncate line it is easier to process later
            	data.loc[line_i,col_i]=field
            	col_i+=1
            	cline=line[:10]
            	if cline=='          ':
            		field=np.nan
            	else:
            		field=float(cline) #' {:7.2f}'
            	line=line[10:] #Truncate line it is easier to process later
            	data.loc[line_i,col_i]=field
            	col_i+=1
            	cline=line[:6]
            	if cline=='      ':
            		field=np.nan
            	else:
            		field=int(cline) #' {:7.2f}'
            	line=line[6:] #Truncate line it is easier to process later
            	data.loc[line_i,col_i]=field
            	col_i+=1
            	cline=line[:7]
            	if cline=='       ':
            		field=np.nan
            	else:
            		field=float(cline) #' {:7.2f}'
            	line=line[7:] #Truncate line it is easier to process later
            	data.loc[line_i,col_i]=field
            	col_i+=1
    #We remove unecessary columns:
    data_2=data.set_index(keys=0)
    data_2.index.name='frame'
    data_2=data_2.dropna(axis=1,how='all')
    data_2.index=data_2.index.astype(int)
    data_2=data_2.astype(float)
    #Calculate the number of ellipses
    assert np.mod(data_2.shape[1],len(name_param_ellipse))==0, "Wrong number of ellipses read from files {} not devisable by {}".format(data_2.shape[1],len(name_param_ellipse))
    #Name nicely the data
    columns=list()
    for mark_i in range(data_2.shape[1]//len(name_param_ellipse)):
        for label in name_param_ellipse:
            columns.append((mark_i,label))
    data_2.columns=pd.MultiIndex.from_tuples(columns)
    data_2.columns.names=['marker','ellipse_param']
    data_2.head()
    return data_2

def get_nbmarker(data):
	return len(data.columns.levels[0])

def get_ellipse_param():
	return ['x','y','orientation','size','roundness']

def write_tra_file(traFile, data, mode='w'):
    """
     write_tra - write trajectory data into an ivtrace file
       write_tra (traFile, traData) writes the trajectory data in traData into
       an ascii file reconstructing the format ivTrace writes.
    
       :param traFile: filename of the trajectory
       :type traFile: string
       :param traData: Dataframe with index the frame number
       :type traData: pandas dataframe
       
       >>write_tra_file('trajectory.tra', data);
    """
    #Convert dataframe to matrix
    nb_markers=get_nbmarker(data)
    name_param_ellipse=get_ellipse_param()
    #Save
    file=open(traFile, mode, newline='\n');
    for frame_i in data.index.values:
        #print index field of the line
        file.write('{:7d}'.format(int(frame_i)))
        #for each region write 5 fields...
        for mark_i in range(nb_markers):
            val=data.loc[frame_i, (mark_i,name_param_ellipse[0])] #x
            if np.isnan(val):
            	file.write('        ')
            else:
            	file.write(' {:7.2f}'.format(val))
            
            val=data.loc[frame_i, (mark_i,name_param_ellipse[1])] #y
            if np.isnan(val):
            	file.write('        ')
            else:
            	file.write(' {:7.2f}'.format(val))
            
            val=data.loc[frame_i, (mark_i,name_param_ellipse[2])] #orientation
            if np.isnan(val):
            	file.write('          ')
            else:
            	file.write(' {:9.5f}'.format(val))
            
            val=data.loc[frame_i, (mark_i,name_param_ellipse[3])] #size
            if np.isnan(val):
            	file.write('      ')
            else:
            	file.write(' {:5d}'.format(int(val)))
            
            val=data.loc[frame_i, (mark_i,name_param_ellipse[4])] #roundness
            if np.isnan(val):
            	file.write('       ')
            else:
            	file.write(' {:6.2f}'.format(val))
        # append end of line
        file.write('\n');
    file.close()



