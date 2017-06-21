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
    Mat plateImg;
    vector<Point> plateOutline;

    tightPreprocessing(img);
    hardSegmentation(img, plateOutline);
    phatPerspectiveNormalizer(img, plateOutline, plateImg);
    return megaPlateRecognisation(plateImg);
}

void camPath(int camNo){
    CvCapture* capture = 0;
    capture = cvCaptureFromCAM( camNo );
    if(!capture){
    	cout << "No valid camera " << endl;
    	return;
    }
    Mat display, frame;
    int plateId = -1;
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
		}else{
		   if(debug) cout << "No known Plate found." << endl;
		}
        if( waitKey( 10 ) >= 0 ){
            cvReleaseCapture( &capture );
            break;
        }
    }
}

void imagePath(string path){
	Mat display = imread( path, CV_LOAD_IMAGE_COLOR );
	showScaled(windowname, display);
	waitKey();
}

int main( int argc, char** argv )
{
    if (argc==1) {
       cerr << "Usage" << endl;
       cerr << "dices [debug] {[image], cam [id]}" << endl;
       return 1;
    }
    int imageCtr = 1;

    if(!strcmp(argv[1], "debug")){
    	debug = true;
    	imageCtr++;
    	cout << "Debug mode active." << endl;
    }


    if(!strcmp(argv[imageCtr], "cam")){
		if(debug) namedWindow( windowname, WINDOW_AUTOSIZE );
    	int camNo = -1;
    	if(argc > imageCtr+1)
    		camNo = atoi(argv[imageCtr+1]);
    	cout << "Grepping cam " << camNo << endl;
    	 //0=default, -1=any camera, 1..99=your camera
    	camPath(camNo);
    }else{
    	namedWindow( windowname, WINDOW_AUTOSIZE );
    	if(argc > imageCtr+1){
    		imagePath(argv[imageCtr+1]);
    	}else{
    		cout << "No image given!" << endl;
    	}
    }
}
