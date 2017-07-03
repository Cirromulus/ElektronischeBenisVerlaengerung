#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <algorithm>
#include <iostream>

using namespace std;
using namespace cv;

// The images are very large, hence for displaying they are scaled down to
// the size below.
int windowWidth = 1200, windowHeight = 600;



string windowname = "ACV My own Project"; //TODO: Change

//! A class for a general point found somewhere.
//! It can display some text and/or a cross as a result.
/*! You can extend this class to hold your result of whatever you have
    recognized if that fits well.
 */
class Result {
  public:
    //! Where is the Result in the image?
    //! This can either be a point, i.e. imageRegion.size=Size(0,0)
    //! Or a real rectangle
    RotatedRect imageRegion; 
    //! Description of the point found (written centered on the image)
    string text;
    //! Whether to paint a cross at p
    bool doShowCross;
    //! In which color to paint the result (default red)
    Scalar color;


  Result () :doShowCross(false) {}
  Result (RotatedRect imageRegion, string text, bool doShowCross=true, Scalar color=CV_RGB(0,255,0))
    :imageRegion(imageRegion), text(text), doShowCross(doShowCross), color(color) {}
  Result (Point p, string text, bool doShowCross=true, Scalar color=CV_RGB(0,255,0)) 
    :imageRegion (p,Size(),0), text(text), doShowCross(doShowCross), color(color) {} 
  Result (int x, int y, string text, bool doShowCross=true, Scalar color=CV_RGB(0,255,0)) 
    :imageRegion (Point(x,y),Size(),0), text(text), doShowCross(doShowCross), color(color) {} 
};

// Scales display down to windowWidth*windowHeight and shows it in
// windowName.
void showScaled (string windowName, Mat& display) {
   int wx, wy;
   if (display.cols<=windowWidth && display.rows<=windowHeight) {
      wx = display.cols;
      wy = display.rows;
   }
   else {
      wx = (int) std::min ((float) windowWidth, display.cols*(((float)windowHeight)/display.rows));
      wy = (int) std::min ((float) windowHeight, display.rows*((float)windowWidth/display.cols));
   }
   cout << "Scaled " << display.cols << "*" << display.rows << " to " << wx << "*" << wy << endl;
   Mat scaledDisplay(wy, wx, CV_8UC3, Scalar(0,0,0));
   resize (display, scaledDisplay, Size(wx,wy), 0,0, INTER_AREA);
   imshow (windowName, scaledDisplay);
}

// Draws a cross at p of given size (width=height) and color
void drawCross (Mat& display, Point p, int size, Scalar color) {
   line (display, Point (p.x-size/2, p.y-size/2), Point (p.x+size/2, p.y+size/2), color, 3);
   line (display, Point (p.x-size/2, p.y+size/2), Point (p.x+size/2, p.y-size/2), color, 3);
}

//! Draws a single result on display
//! result.text is written with its center at  result.imageRegion.center
//! If result.doShowCross a cross is painted at result.imageRegion.center
//! If result.imageRegion has non-empty size, a it is drawn a a rotated rectangle
//! Everything is shown in result.color
void draw (Mat& display, const Result& result) {
  Scalar color = result.color;
  if (result.imageRegion.size.width>0 || result.imageRegion.size.height>0) {
    Point2f vertices[4];
    result.imageRegion.points(vertices);
    for (int i = 0; i < 4; i++)
       line(display, vertices[i], vertices[(i+1)%4], color, 3);
  }
  if (!result.text.empty()) {
    Size textSize = getTextSize (result.text, FONT_HERSHEY_PLAIN, 8, 4, NULL);
    Point textPos = result.imageRegion.center;
    textPos.x -= textSize.width/2;
    textPos.y += textSize.height/2;
    putText (display, result.text, textPos, FONT_HERSHEY_PLAIN, 8, color, 4);
  }
  if (result.doShowCross)
    drawCross (display, result.imageRegion.center, 30, color);
}

//! Draws a list of Results on the image display
void draw (Mat& display, const vector<Result>& Results) {
   for (int i=0; i<Results.size(); i++) draw (display, Results[i]);
}

// Makes a RGB image darker by 1/4
void makeImageDarker (Mat& display) {
   for (int y=0; y<display.rows; y++) for (int x=0; x<display.cols; x++) 
      for (int c=0; c<3; c++) {
      display.at<Vec3b>(y,x)[c] = (display.at<Vec3b>(y,x)[c] * 3)/4;
   }
}

ostream& operator<< (std::ostream& stream, const RotatedRect r) {
   stream << r.center.x << "," << r.center.y;
   if (r.size.width>0 || r.size.height>0) 
      stream << "(" <<r.size.width << "*" << r.size.height << "r" << r.angle << ")";
   return stream;
}

//! Prints information about one result found
ostream& operator<< (std::ostream& stream, const Result& result) {
   stream << result.text << "@" << result.imageRegion <<" ";
   return stream;
}

//! Prints information about a list of results found
ostream& operator<< (std::ostream& stream, const vector<Result>& results) {
   for (int i=0; i<results.size(); i++) stream << results[i] << ends;
   stream << endl;
   return stream;
}



/********** BELOW HERE TODOs ******************************/

//! Recognizes all objects in images and returns them in results
//! display is passed, just in case you want to show something
//! for debugging reasons
void findObjects (Mat& image, Mat& display, vector<Result>& results) {
   // TODO: Implement
   results.clear();
   results.push_back (Result(100,100,"",true));
   results.push_back (Result(Point(image.cols/2, image.rows/2), "Mitte", true));
   results.push_back (Result(RotatedRect(Point(600, 400), Size(300,150), 30), "schraeg", true, CV_RGB(255,0,0)));
}



/***************************************************************************/

int main( int argc, char** argv )
{
    if (argc==1) {
       cerr << "Usage" << endl;
       cerr << "AcvProject [image] ... [image]" <<endl;
       return 1;
    }
    int imageCtr = 0;
    namedWindow( windowname, WINDOW_AUTOSIZE );
    vector<int> statistics(7); // Entries 1..6, 0 is not needed
    while (imageCtr+1<argc) {
       /// Load the source image
       string filename;
       if (0<=imageCtr && imageCtr+1<argc) filename = argv[imageCtr+1];
       else filename="";
       Mat src = imread( filename, CV_LOAD_IMAGE_COLOR );
       Mat display = imread( filename, CV_LOAD_IMAGE_COLOR );
       makeImageDarker (display);


       vector<Result> results;
       findObjects (src, display, results);
       cout << "Found the following objects: "<< endl << results << endl;
       draw (display, results);

       showScaled (windowname, display);
       if (waitKey()==27) break;
       imageCtr++;
    } 
    cout << endl << endl;
}
