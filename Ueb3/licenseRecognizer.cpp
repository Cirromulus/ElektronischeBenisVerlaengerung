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

bool debug = false;

string windowname = "License Recognization PLATOS";

/***************************************************************************/

int main( int argc, char** argv )
{
    if (argc==1) {
       cerr << "Usage" << endl;
       cerr << "dices [debug] [image] ... [image]" << endl;
       return 1;
    }
    int imageCtr = 0;
    if(!strcmp(argv[1], "debug")){
    	debug = true;
    	imageCtr++;
    	cout << "Debug mode active." << endl;
    }
    namedWindow( windowname, WINDOW_AUTOSIZE );
    vector<int> statistics(7); // Entries 1..6, 0 is not needed
    while (imageCtr+1<argc) {
       /// Load the source image
       string filename;
       if (0<=imageCtr && imageCtr+1<argc) filename = argv[imageCtr+1];
       else filename="";
       Mat display = imread( filename, CV_LOAD_IMAGE_COLOR );
       showScaled(windowname, display);
       Mat plateImg;
       int plateId;
       vector<Point> plateOutline;

       tightPreprocessing(display);
       hardSegmentation(display, plateOutline);
       phatPerspectiveNormalizer(display, plateOutline, plateImg);
       if((plateId = megaPlateRecognisation(plateImg)) > 0){
    	   cout << "Found Plate " << string(knownPlates[plateId]) << endl;
       }else{
    	   cout << "No known Plate found." << endl;
       }

       if (waitKey()==27) break;
       imageCtr++;
    } 
}
