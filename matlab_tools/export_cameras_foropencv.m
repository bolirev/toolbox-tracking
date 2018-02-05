function export_cameras_foropencv(cameraParams, filename)
    docNode = com.mathworks.xml.XMLUtils.createDocument('opencv_storage');
    opencv_storage = docNode.getDocumentElement;
    
    element_name = 'rotation';    
    element_data = cameraParams.RotationOfCamera2;
    element = create_element(docNode,element_name, element_data);
    opencv_storage.appendChild(element)
    
    element_name = 'translation';    
    element_data = cameraParams.TranslationOfCamera2;
    element = create_element(docNode,element_name, element_data);
    opencv_storage.appendChild(element)
    
    element_name = 'fundamental_matrix';    
    element_data = cameraParams.FundamentalMatrix;
    element = create_element(docNode,element_name, element_data);
    opencv_storage.appendChild(element)
    
    element_name = 'essential_matrix';    
    element_data = cameraParams.EssentialMatrix;
    element = create_element(docNode,element_name, element_data);
    opencv_storage.appendChild(element)
    
    camnames = {'CameraParameters1' ,'CameraParameters2'};
    for cam_i = 0:(length(camnames)-1)
        camname = camnames{cam_i+1};
        element_name = sprintf('intrinsic_matrix_%d',cam_i);
        element_data = cameraParams.(camname).IntrinsicMatrix;
        element = create_element(docNode,element_name, element_data);
        opencv_storage.appendChild(element)

        element_name = sprintf('distortion_%d',cam_i);
        rdist = cameraParams.(camname).RadialDistortion;
        tdist = cameraParams.(camname).TangentialDistortion;
        if length(rdist)==3
            dist = [rdist(1:2) tdist rdist(3)];
        elseif length(rdist)==2
            dist = [rdist(1:2) tdist];
        else
            error('Unsorported number of radial distortion')
        end
        element_data = dist;
        element = create_element(docNode,element_name, element_data);
        opencv_storage.appendChild(element)
    end
    
    xmlwrite(filename,docNode);
end

function element = create_element(docNode,element_name, data)
    element = docNode.createElement(element_name);
    element.setAttribute('type_id','opencv-matrix')
    rows = docNode.createElement('rows');
    rows.appendChild(docNode.createTextNode(num2str(size(data,1))))
    element.appendChild(rows)
    
    cols = docNode.createElement('cols');
    cols.appendChild(docNode.createTextNode(num2str(size(data,2))))
    element.appendChild(cols)
    
    dt = docNode.createElement('dt');
    dt.appendChild(docNode.createTextNode('d'))
    element.appendChild(dt)
    
    data_str = '';
    fdata = reshape(data,1,[]);
    for data_i = 1:length(fdata)
        data_str=strcat(data_str, sprintf("%0.18e",fdata(data_i)));
        data_str=strcat(data_str," ");
    end
    data_str=sprintf("\n%s",data_str);    
    data = docNode.createElement('data');
    data.appendChild(docNode.createTextNode(data_str))
    element.appendChild(data)
end