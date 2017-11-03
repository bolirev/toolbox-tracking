#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "Python.h"
#include "numpy/arrayobject.h"

/* 
TODO:
    * Write Function to get cropped
    * Write Timing functions
*/

#include <stdexcept>
#include <iostream>
#include <string>
#include <limits>


#define BTTRACKER_GPU 0
//Include open cv
#include "opencv2/core.hpp"
#include "opencv2/core/utility.hpp"

//Include open cv
#if BTTRACKER_GPU==1
#include "opencv2/cudalegacy.hpp"
#include "opencv2/cudabgsegm.hpp"
#else
#include "opencv2/bgsegm.hpp"
#endif

#include "opencv2/video.hpp"
#include "opencv2/highgui.hpp"

#include <boost/python.hpp>

//#include "opencv2.cpp"

using namespace std;
using namespace cv;

#if BTTRACKER_GPU==1
using namespace cv::cuda;
#endif

#define PY_ARRAY_UNIQUE_SYMBOL API_ARRAY_API
#include <boost/python.hpp>
//#include <pyboostcvconverter/pyboostcvconverter.hpp>
#include "opencv_swig_python.h"

string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}


class BeeTrackTracker
{
    public:
    enum processing_steps {none,
                       started,
                       read,
                       masked,
                       segmented,
                       eroded,
                       dilated,
                       contours_found,
                       contours_filtered,
                       croped,
                       timing};

    BeeTrackTracker();     // constructor
    void run();
    void set_image_py(PyObject* image);
    void set_mask_py(PyObject* image);
    PyObject* get_data_py(enum processing_steps step);
    //Set parameter
    void set_area_lim(int,int);
    void set_roundness_lim(int,int);
    void set_erode_iter(int);
    void set_dilate_iter(int);
	void set_roi(int, int);
	void set_max_nb_bee(int);
    //
    void set_image_opencv(Mat image);
    void set_mask_opencv(Mat image);
    Mat get_image_data_opencv(enum processing_steps step);

    private:
    #if BTTRACKER_GPU==1
    //Declare input image
        GpuMat orig_image;
        GpuMat mask_image;
        //Declare processed image
        GpuMat masked_im;
        GpuMat segmented_im;
        GpuMat eroded_im;
        GpuMat dilated_im;
    #else
    //Declare input image
        Mat orig_im;
        Mat mask_im;
        //Declare processed image
        Mat masked_im;
        Mat segmented_im;
        Mat eroded_im;
        Mat dilated_im;
    #endif
        
    Mat processed_im;
    //Declare contours
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;

    //Declare croping param
    Rect my_roi;
    Mat ellipse_mask;
    Mat cropped_images;

    //Declare private functions
    void mask_image();
    void segment_image();
    void erode_image();
    void dilate_image();
    void find_contours();
    void filter_contours();
    void crop();

    //Declare init method
    void init_ellipses();
    void init_centers();
	void init_croped();
        
    //Declare filtering method
    Ptr<BackgroundSubtractor> pSegmentation;
    #if BTTRACKER_GPU==1
        Ptr<Filter> erode;
        Ptr<Filter> dilate;
    #endif

    //Declare result variables
    vector<RotatedRect> ellipses;

    //Declare parameters
    int erode_iter;
    int dilate_iter;
    Mat kernel;
    //For ellipse filtering
    double min_area;
    double max_area;
    double min_roundness;
    double max_roundness;
    int max_nb_bee;
    //For background substracktion    
    int initializationFrames;
    //Constant
    const static int rotatedrect_size = 5;
};

/*****************************************************************************\
    INIT
\*****************************************************************************/
BeeTrackTracker::BeeTrackTracker()
{
    erode_iter =2;
    dilate_iter=2;
    kernel=Mat();
    min_area=     0;
    max_area=100000;
    min_roundness=0;
    max_roundness=1;
    max_nb_bee= 5;
    initializationFrames=40;
    //Create Background
    pSegmentation = bgsegm::createBackgroundSubtractorMOG(60); //
    //Create errode/dilate filter
    #if BTTRACKER_GPU==1
    erode  = createMorphologyFilter(cv::MORPH_ERODE,  src.type(), kernel);
    dilate = createMorphologyFilter(cv::MORPH_DILATE, src.type(), kernel);
    #endif
    init_ellipses();
    //init region of interest for croping.
    //the height and width of all region is constant
	set_roi(150,120);
	init_croped();
}
////
void BeeTrackTracker::init_ellipses()
{
    ellipses.clear();
    contours.clear();
    hierarchy.clear();
}
void BeeTrackTracker::init_croped()
{
    //Init crop matrix
    int dim[3];
    dim[0]=max_nb_bee;
    dim[1]=my_roi.height;
    dim[2]=my_roi.width;
    cropped_images=Mat(3, dim, CV_8U, cv::Scalar(0));
}
/*****************************************************************************\
    SET/GET

    1: PYTHON / C++
    2: PYTHON Only
    3: C++ Only

\*****************************************************************************/
/****************************************\
        1: PYTHON / C++
\****************************************/
void BeeTrackTracker::set_area_lim(int min,int max)
{
    min_area=min; 
    max_area=max;
}
void BeeTrackTracker::set_roundness_lim(int min,int max)
{
    min_roundness=min; 
    max_roundness=max;
}
void BeeTrackTracker::set_erode_iter(int iter)
{
    erode_iter=iter; 
}
void BeeTrackTracker::set_dilate_iter(int iter)
{
    dilate_iter=iter; 
}
void BeeTrackTracker::set_roi(int height, int width)
{
    my_roi=Rect();
    my_roi.height=height;
    my_roi.width=width;
    //reset ellipse mask
    ellipse_mask=Mat::zeros(my_roi.height,my_roi.width,CV_8U);
}
void BeeTrackTracker::set_max_nb_bee(int nb_bee)
{
	max_nb_bee=nb_bee;
	init_croped();
}
/****************************************\
        2: PYTHON Only
\****************************************/
void BeeTrackTracker::set_image_py(PyObject* image_py)
{
    Mat image;
    if (! pyopencv_to(image_py, image))
    {
        throw std::invalid_argument( "OpenCV conversion failed to convert to Mat"); 
    }
    //image=pbcvt::fromNDArrayToMat(image_py);
    set_image_opencv(image);
}
void BeeTrackTracker::set_mask_py(PyObject* image_py)
{
    Mat image;
    if (! pyopencv_to(image_py, image))
    {
        throw std::invalid_argument("OpenCV conversion failed to convert to Mat"); 
    }
    //image=pbcvt::fromNDArrayToMat(image_py);
    set_mask_opencv(image);
}

PyObject* BeeTrackTracker::get_data_py(enum processing_steps step)
{
  switch(step)
  {
      case read:
      case masked:
      case segmented:
      case eroded:
      case dilated:
	  case croped:
      {
    Mat toreturn=get_image_data_opencv(step);
    return pyopencv_from(toreturn);
      }
      case contours_filtered:
      {
        const int nb_ellipses = ellipses.size();
    //cout<<"nb ellipses:  "<<nb_ellipses<<endl;
    int planSize[] = {nb_ellipses,rotatedrect_size};
    Mat toreturn(2,planSize,CV_64F);
    for (int i = 0; i<toreturn.size[0]; i ++)
    {
        toreturn.at<double>(i, 0)=ellipses[i].center.x; //along column
        toreturn.at<double>(i, 1)=ellipses[i].center.y; //along row
        toreturn.at<double>(i, 2)=ellipses[i].size.height;
        toreturn.at<double>(i, 3)=ellipses[i].size.width;
        toreturn.at<double>(i, 4)=ellipses[i].angle;
    }
    return pyopencv_from(toreturn);
       }
       default : throw std::invalid_argument( "get_data receive an invalid step" );
    }
}
/****************************************\
        3: C++ Only
\****************************************/
void BeeTrackTracker::set_image_opencv(Mat image)
{
    if(image.size()!=mask_im.size())
    {
        throw std::invalid_argument( "image and mask should have the same size.\n"
                         "Have you already set the mask?" );
    }
    #if BTTRACKER_GPU==1
    if(masked_im.size()!=image.size())
    {
         masked_im.upload(image);
         masked_im.setTo(0);
    }
    if(segmented_im.size()!=image.size()) segmented_im.upload(0*image);
    if(eroded_im.size()!=image.size()) eroded_im.upload(0*image);
    if(dilated_im.size()!=image.size()) dilated_im.upload(0*image);
    orig_im.upload(image);
    #else
    if(masked_im.size()!=image.size()){
        masked_im=image.clone();
        masked_im.setTo(0);
    }
    if(segmented_im.size()!=image.size()) segmented_im=0*image.clone();
    if(eroded_im.size()!=image.size()) eroded_im=0*image.clone();
    if(dilated_im.size()!=image.size()) dilated_im=0*image.clone();
    orig_im=image;
    #endif
}
void BeeTrackTracker::set_mask_opencv(Mat image)
{
    #if BTTRACKER_GPU==1
    mask_im.upload(image);
    #else
    mask_im=image;
    #endif
    
}

Mat BeeTrackTracker::get_image_data_opencv(enum processing_steps step)
{
    Mat toreturn;
    #if BTTRACKER_GPU==1
    switch(step)
    {
        case read: orig_im.download(toreturn); break;
        case masked: masked_im.download(toreturn); break;
        case segmented: segmented_im.download(toreturn); break;
        case eroded: eroded_im.download(toreturn); break;
        case dilated: dilated_im.download(toreturn); break;
        default : throw std::invalid_argument( "get_image_data_opencv receive an invalid step" );
    }
    #else
    switch(step)
    {
        case read:    toreturn=orig_im; break;
        case masked:    toreturn=masked_im; break;
        case segmented: toreturn=segmented_im; break;
        case eroded:  toreturn=eroded_im; break;
        case dilated:  toreturn=dilated_im; break;
        case croped:
        {
			crop();
			toreturn=cropped_images; break;
		}
        default : throw std::invalid_argument( "get_image_data_opencv receive an invalid step" );
    }
    #endif
    return toreturn;
}

/*****************************************************************************\
    
    Computation

\*****************************************************************************/

void BeeTrackTracker::run()
{
    mask_image();
    segment_image();
    erode_image();
    dilate_image();
    #if BTTRACKER_GPU==1
    dilated_im.download(processed_im);
    #else
    dilated_im.copyTo(processed_im,mask_im);
    #endif
    find_contours();
    filter_contours();
    crop();
}
void BeeTrackTracker::mask_image()
{
    orig_im.copyTo(masked_im,mask_im);
}
void BeeTrackTracker::segment_image()
{
    #if BTTRACKER_GPU==1
    pSegmentation.operator()(masked_im, segmented_im,-1);
    #else
    pSegmentation->apply(masked_im, segmented_im,-1);
    #endif
    threshold(segmented_im, segmented_im, 10,255,THRESH_BINARY);
}
void BeeTrackTracker::erode_image()
{
    segmented_im.copyTo(eroded_im,mask_im);
    for(int i=0; i<erode_iter; i++)
    {
        #if BTTRACKER_GPU==1
        erode->apply(eroded_im, eroded_im);
        #else
        erode(eroded_im, eroded_im, kernel);
        #endif
    }
}
void BeeTrackTracker::dilate_image()
{
    eroded_im.copyTo(dilated_im,mask_im);
    for(int i=0; i<dilate_iter; i++)
    {
        #if BTTRACKER_GPU==1
        dilate->apply(dilated_im, dilated_im);
        #else
        dilate(dilated_im, dilated_im, kernel);
        #endif
    }
}
void BeeTrackTracker::find_contours()
{
    findContours( processed_im,
          contours,
          hierarchy,
          RETR_TREE,
          CHAIN_APPROX_SIMPLE,
          Point(0, 0) );
}
void BeeTrackTracker::filter_contours()
{
    //cout<<contours.size()<<endl;
    //We are going to filter a bit our contours
    ellipses.clear();
    RotatedRect curr_ellipse;
    cout<<"erh vjkewhrkcjhcrw: "<<endl;
    for( unsigned int cont_k = 0; cont_k< contours.size(); cont_k++ )
    {
        //Filter by number of points
        //Because we want to filter an ellipse
        //we need at list 5 points
        bool acceptable=(contours[cont_k].size()>=5);
        //cout<<"acceptable: "<<acceptable<<endl;
        if(acceptable==false){continue;};
        //Area
        double cont_area=contourArea(contours[cont_k]);
        acceptable=((cont_area>min_area) && 
                    (cont_area<max_area));
        cout<<"cont_area: "<<cont_area<<"\t acceptable: "<<acceptable<<endl;
        if(acceptable==false){continue;};
        //Fit ellipse
        curr_ellipse=fitEllipse(contours[cont_k]);
        //Roundness
        //Major axis / Minor axis
        double roundness=curr_ellipse.size.width/(1.0*curr_ellipse.size.height);
        acceptable=((roundness>min_roundness) && 
                    (roundness<max_roundness));
        //cout<<"roundness: "<<roundness<<"\t acceptable: "<<acceptable<<endl;
        if(acceptable==false){continue;};
        
        //We have a winner, the curr_ellipse
        //is acceptable
        ellipses.push_back(curr_ellipse);
    }
}
void BeeTrackTracker::crop()
{
  Mat cropped_im;
  cropped_images.setTo(0);
  for( unsigned int ell_k = 0; ell_k< ellipses.size(); ell_k++ )
  {
    //reset ellipse mask to zero
    ellipse_mask = Scalar::all(0);
    //Assign roi corner x,y
    my_roi.x=ellipses[ell_k].center.x-my_roi.width/2;
    my_roi.y=ellipses[ell_k].center.y-my_roi.height/2;
    //Since roi may be much larger than ellipse, apply mask based on ellipse
    ellipse(ellipse_mask,
        Point(my_roi.width/2,my_roi.height/2), //Center of roi
        Size(ellipses[ell_k].size.width/2,ellipses[ell_k].size.height/2), //Ellipse half axes
        ellipses[ell_k].angle,0,360,255,-1);
    
    // Crop the full image to that image contained by the rectangle myROI
    orig_im(my_roi).copyTo(cropped_im,ellipse_mask);
    //copy to big data
    for(int row=0; row<cropped_im.rows; row++)
    {
      for(int col=0; col<cropped_im.cols; col++)
    	{cropped_images.at<uchar>(ell_k,row,col)=cropped_im.at<uchar>(row,col);}
    }
    
   }
}

/*****************************************************************************\






\*****************************************************************************/
#include <boost/python.hpp>
using namespace boost::python;

BOOST_PYTHON_MODULE(beetracktracker)
{
    // C++ <-> numpy <-> opencv convertion requires this
    initOpenCVSwig();

    scope the_scope=
        class_<BeeTrackTracker>("beetracktracker", init<>())
            .def(init<>())
            .def("set_area_lim",&BeeTrackTracker::set_area_lim)
            .def("set_roundness_lim",&BeeTrackTracker::set_roundness_lim)
            .def("set_erode_iter",&BeeTrackTracker::set_erode_iter)
            .def("set_dilate_iter",&BeeTrackTracker::set_dilate_iter)
			.def("set_roi",&BeeTrackTracker::set_roi)
            .def("set_max_nb_bee",&BeeTrackTracker::set_max_nb_bee)
            .def("run",&BeeTrackTracker::run)
            .def("set_image",&BeeTrackTracker::set_image_py)
            .def("set_mask", &BeeTrackTracker::set_mask_py)
            .def("get_data", &BeeTrackTracker::get_data_py)
        ;
    enum_<BeeTrackTracker::processing_steps>("processing_steps")
        .value("none",    BeeTrackTracker::none)
        .value("started", BeeTrackTracker::started)
        .value("read",    BeeTrackTracker::read)
        .value("masked",  BeeTrackTracker::masked)
        .value("segmented", BeeTrackTracker::segmented)
        .value("eroded",  BeeTrackTracker::eroded)
        .value("dilated", BeeTrackTracker::dilated)
        .value("contours_found", BeeTrackTracker::contours_found)
        .value("contours_filtered", BeeTrackTracker::contours_filtered)
        .value("croped", BeeTrackTracker::croped)
        .value("timing", BeeTrackTracker::timing)
        .export_values()
        ;
}


