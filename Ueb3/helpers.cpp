/*
 * helpers.cpp
 *
 *  Created on: Jun 20, 2017
 *      Author: rooty
 */

#include "helpers.hpp"

using namespace std;
using namespace cv;

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
   //cout << "Scaled " << display.cols << "*" << display.rows << " to " << wx << "*" << wy << endl;
   Mat scaledDisplay(wy, wx, CV_8UC3, Scalar(0,0,0));
   resize (display, scaledDisplay, Size(wx,wy), 0,0, INTER_AREA);
   imshow (windowName, scaledDisplay);
}

void drawHull(Mat& dst, vector<Point2i> hull, int col){
	for (unsigned int i = 0; i < hull.size(); i++){
		line(dst, hull[i], hull[(i+1)%hull.size()], color[col%6]);
	}
}

/**
 * @brief draws a vector of points in given size in cycling colors
 */
void drawApprox(Mat& dst, vector<Point> hull, int col, int size){
	for (unsigned int i = 0; i < hull.size(); i++){
		line(dst, hull[i], hull[(i+1)%hull.size()], color[col%6], size);
	}
}

/**
 * @brief draws a vector of vector of points in given size in cycling colors
 */
void drawApproxes(Mat& dst, vector<vector<Point> > app, int size){
	for (unsigned int i = 0; i < app.size(); i++){
		drawApprox(dst, app[i], i, size);
	}
}

void drawRect(Mat& dst, RotatedRect& rec, int i) {
  Scalar myColor = color[i%6];
  Point2f vertices[4];
  rec.points(vertices);
  for (int i = 0; i < 4; i++)
    line(dst, vertices[i], vertices[(i+1)%4], myColor, 8);
}

void drawRects(Mat& dst, vector<RotatedRect>& rects){
	int i = 0;
	for(auto rect : rects){
		drawRect(dst, rect, i++);
	}
}

// Draws a cross at p of given size (width=height) and color
void drawCross (Mat& display, Point p, int size, Scalar color) {
   line (display, Point (p.x-size/2, p.y-size/2), Point (p.x+size/2, p.y+size/2), color, 3);
   line (display, Point (p.x-size/2, p.y+size/2), Point (p.x+size/2, p.y-size/2), color, 3);
}

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

// Makes a RGB image darker by 1/4
void makeImageDarker (Mat& display) {
   for (int y=0; y<display.rows; y++) for (int x=0; x<display.cols; x++)
      for (int c=0; c<3; c++) {
      display.at<Vec3b>(y,x)[c] = (display.at<Vec3b>(y,x)[c] * 3)/4;
   }
}

/**
 * @brief Adjusts contrast and brightness to the MAX
 */
void preprocessColors(Mat &img, double highestValue){
	double min, max;
	minMaxLoc(img, &min, &max);
	double factor = highestValue / (max - min);
	Vec3b offs(-min, -min, -min);
    for( int y = 0; y < img.rows; y++ ) {
        for( int x = 0; x < img.cols; x++ ) {
                img.at<Vec3b>(y,x) =  factor*(img.at<Vec3b>(y,x)) + offs;	//screw that
        }
    }
}
