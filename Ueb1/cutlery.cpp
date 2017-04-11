// Implement a solution to localize and identify the objects in the image
// from the set of possible objects (plate, fork, knife, spoon). The position
// where to add code is marked by "TODO".
// Consider
// http://docs.opencv.org/3.1.0/d3/dc0/group__imgproc__shape.html

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;



string windowname = "Cutlery Detection Software";

class Item {
  public:
    //! Which object?
    string name;
    // It locations (position, extension and orientation) in the image
    RotatedRect imageRegion; 
    // A running index yielding a color  
    int colorIndex;

  Item (string name, RotatedRect imageRegion, int colorIndex=0)
     :name(name), imageRegion(imageRegion), colorIndex(colorIndex)
  {} 
};

void draw (Mat& dst, const Item& item) {
  Scalar color[6] = 
    {
      CV_RGB (255, 0, 0), CV_RGB (0, 255, 0), CV_RGB (255, 255, 0), CV_RGB (0, 0, 255), CV_RGB (255, 0, 255), CV_RGB (0, 255, 255)
    };
  int idx = item.colorIndex%6;
  if (idx<0) idx+=6;
  Scalar myColor = color[idx];
  Point2f vertices[4];
  item.imageRegion.points(vertices);
  for (int i = 0; i < 4; i++)
    line(dst, vertices[i], vertices[(i+1)%4], myColor);
  int baseLine;
  Size textSize = getTextSize (item.name, FONT_HERSHEY_PLAIN, 2, 1, &baseLine);
  circle (dst, item.imageRegion.center, 2, myColor);
  Point textPos = item.imageRegion.center;
  textPos.x -= textSize.width/2;
  putText (dst, item.name, textPos, FONT_HERSHEY_PLAIN, 2, myColor, 1);
}

void draw (Mat& dst, const vector<Item>& items) {
   for (int i=0; i<items.size(); i++) draw (dst, items[i]);
}

int main( int argc, char** argv )
{
    int imageCtr = 0;
    namedWindow( windowname, WINDOW_AUTOSIZE );
    while (imageCtr+1<argc) {
       /// Load the source image
       string filename;
       if (0<=imageCtr && imageCtr+1<argc) filename = argv[imageCtr+1];
       else filename="";
       Mat src = imread( filename, CV_LOAD_IMAGE_GRAYSCALE );
       Mat display;
       cvtColor(src, display, CV_GRAY2BGR);


       vector<Item> items;
// TODO: Implement here code to localize and identify the items on the image
// Below is dummy code to show, how an item object is generated
       items.push_back(Item("plate", RotatedRect(Point(500,400),Size(200,100), 45), 0));
       
       draw (display, items);
       imshow(windowname, display);
       if (waitKey()==27) return 1;
       imageCtr++;
    } 
    return 0;
}