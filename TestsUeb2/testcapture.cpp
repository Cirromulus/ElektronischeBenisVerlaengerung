#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;


int main( int argc, const char** argv )
{
    CvCapture* capture = 0;
    Mat frame, frameCopy, image;

    int cam = CV_CAP_ANY;
    if(argc > 1)
	cam = atoi(argv[1]);
    capture = cvCaptureFromCAM( cam ); //0=default, -1=any camera, 1..99=your camera
    if(!capture) cout << "No camera detected" << endl;

    cvNamedWindow( "result", CV_WINDOW_AUTOSIZE );

    if( capture )
    {
        cout << "In capture ..." << endl;
        for(;;)
        {
            IplImage* iplImg = cvQueryFrame( capture );
            frame = cvarrToMat(iplImg);
            if( frame.empty() )
                break;
            if( iplImg->origin == IPL_ORIGIN_TL )
                frame.copyTo( frameCopy );
            else
                flip( frame, frameCopy, 0 );

	    cvShowImage( "result", iplImg);

            if( waitKey( 10 ) >= 0 ){
                cvReleaseCapture( &capture );
		break;
	    }
        }

        waitKey(0);

    cvDestroyWindow("result");

    return 0;
    }
}
