// Implement a solution to localize and identify the objects in the image
// from the set of possible objects (plate, fork, knife, spoon). The position
// where to add code is marked by "TODO".
// Consider
// http://docs.opencv.org/3.1.0/d3/dc0/group__imgproc__shape.html

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdio.h>
#include <iostream>
#include <vector>

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


       Mat canny_output;
       vector<vector<Point> > contours;
       vector<Vec4i> hierarchy;
       vector<Item> items;

       /// Detect edges using canny
       Canny(src, canny_output, 100, 255);
       /// Find contours
       findContours( canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
       for( int i = 0; i< contours.size(); i++ ){
    	   vector<Point> approx;
    	   approxPolyDP(Mat(contours[i]), approx, 1, true);
    	   //printf("Found a %zu-shape\n", approx.size());
    	   RotatedRect elem = minAreaRect(approx);
    	   String name;
    	   if(approx.size() < 10){
    		   cout << "Too few corners (" << approx.size() << ")" << endl;
    		   continue;
    	   }

    	   if(elem.size.width < 25 || elem.size.height < 25){
    		   cout << "Kantenl'nge too small (" << elem.size.width << "," << elem.size.height << ")" << endl;
			   continue;
    	   }

    	   if(elem.size.width * elem.size.height < 5000){
    		   cout << "Area too small (" << elem.size.width << "," << elem.size.height << ")" << endl;
			   continue;
    	   }

    	   vector<Point> hull;
    	   vector<int>   hullI;
    	   vector<Vec4i> defects;

    	   convexHull(approx, hull);
    	   convexHull(approx, hullI);

    	   cout << "Convex hull: " << hull.size() << endl;
    	   cout << "Convex hullI: " << hullI.size() << endl;

    	   if(hull.size() > 3){
    		   convexityDefects(approx, hullI, defects);
    		   cout << "Defects: " << defects.size() << endl;
			   int a = 0, b = 0;
    		   switch(defects.size()){
    		   case 0:
    			   //assume circle
    			   name += "Plate";
    			   break;
    		   case 9:
    		   case 11:
				   name += "Fork";
				   break;
    		   case 2:
    			   cout << "Knife or spoon" << endl;

    			   if(elem.size.width > elem.size.height){
    				   a = elem.size.width;
					   b = elem.size.height;
    			   }else{
    				   b = elem.size.width;
    				   a = elem.size.height;
    			   }
    			   if(a / b > 6){
    				   name =+ "Knife";
    			   }else{
    				   name =+ "Spoon";
    			   }
    			   break;
    		   default:
    			   name += "Weird";
    			   name += " " + to_string(defects.size());
    		   }

    	   }else{
    		   cout << "no Defects calculateabaele." << endl;
    		   name += "Weird";
    	   }


    	   items.push_back(Item(name.c_str(), elem, i));
       }

       
       draw (display, items);
       imshow(windowname, display);
       if (waitKey()==27) return 1;
       imageCtr++;
    } 
    return 0;
}
