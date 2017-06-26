#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <algorithm>
#include <iostream>
#include <unistd.h>
#include "helpers.hpp"
#include "hotShit.hpp"
#include "knownPlates.hpp"

using namespace std;
using namespace cv;

string windowname = "License Recognization PLATOS";

bool debug = false;

/***************************************************************************/

int pipelineDetect(Mat &img){
    vector<Point> plateOutline;

    tightPreprocessing(img);
    hardSegmentation(img, plateOutline);
    Mat plateImg = phatPerspectiveNormalizer(img, plateOutline);
    return megaPlateRecognisificationessing(plateImg);
}

bool camPath(int camNo){
    CvCapture* capture = 0;
    capture = cvCaptureFromCAM( camNo );
    if(!capture){
    	cout << "No valid camera " << endl;
    	return false;
    }
    Mat display, frame;
    int plateId = -1;
    bool foundAnyPlate = false;
    while(capture){
		IplImage* iplImg = cvQueryFrame( capture );
		frame = cvarrToMat(iplImg);
		if( frame.empty() )
		   break;
		if( iplImg->origin == IPL_ORIGIN_TL )
		   frame.copyTo( display );
		else
		   flip( frame, display, 0 );

		if(debug) cvShowImage( windowname.c_str(), iplImg);

		plateId = pipelineDetect(display);
		if(plateId >= 0){
		   cout << "Found Plate " << string(knownPlates[plateId]) << endl;
		   foundAnyPlate = true;
		}else{
		   if(debug) cout << "No known Plate found." << endl;
		}
        if( waitKey( 10 ) >= 0 ){
            cvReleaseCapture( &capture );
            break;
        }
    }
    return foundAnyPlate;
}

bool imagePath(string path){
	Mat display = imread( path, CV_LOAD_IMAGE_COLOR );
	if(debug) showScaled(windowname, display);
	int plateId = pipelineDetect(display);
	if(plateId >= 0){
	   cout << "Found Plate " << string(knownPlates[plateId]) << endl;
	}else{
	   if(debug) cout << "No known Plate found." << endl;
	}
	if(debug) waitKey();
	return plateId >= 0;
}

int main( int argc, char** argv )
{
    if (argc==1) {
       cerr << "Usage" << endl;
       cerr << argv[0] << " [debug] {[image], cam [id]}" << endl;
       return 1;
    }
    int imageCtr = 1;

    if(!strcmp(argv[1], "debug")){
    	debug = true;
    	imageCtr++;
    	cout << "Debug mode active." << endl;
    }

	if(debug) namedWindow( windowname, WINDOW_AUTOSIZE );

    if(!strcmp(argv[imageCtr], "cam")){
    	int camNo = -1;
    	imageCtr++;
    	if(argc > imageCtr)
    		camNo = atoi(argv[imageCtr]);
    	if(debug) cout << "Grepping cam " << camNo << endl;
    	 //0=default, -1=any camera, 1..99=your camera
    	return camPath(camNo) ? 0 : -1;
    }else{
    	if(argc <= imageCtr){
    		cout << "No image given!" << endl;
    		return -1;
    	}
		return imagePath(argv[imageCtr]) ? 0 : -1;;
    }
}
