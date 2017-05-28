// Implement a solution to localize and identify the objects in the image
// from the set of possible objects (plate, fork, knife, spoon). The position
// where to add code is marked by "TODO".
// Consider
// http://docs.opencv.org/3.1.0/d3/dc0/group__imgproc__shape.html

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <algorithm>
#include <iostream>
#include <unistd.h>

using namespace std;
using namespace cv;

// The images are very large, hence for displaying they are scaled down to
// the size below.
int windowWidth = 1200, windowHeight = 600;

bool debug = false;

string windowname = "Dice Production QA";

class Dice {
  public:
    //! Where is the center of the dice in the image?
    Point p;
    //! Which number does the dice show? 
    //! You can use 0 for showing an auxiliary cross for development purposes
    int eyes;

  Dice () :eyes(0) {}
  Dice (Point p, int eyes) :eyes(eyes), p(p) {} 
  Dice (int x, int y, int eyes) : eyes(eyes), p(x, y) {}
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
   //cout << "Scaled " << display.cols << "*" << display.rows << " to " << wx << "*" << wy << endl;
   Mat scaledDisplay(wy, wx, CV_8UC3, Scalar(0,0,0));
   resize (display, scaledDisplay, Size(wx,wy), 0,0, INTER_AREA);
   imshow (windowName, scaledDisplay);
}

Scalar color[] =
  {
    CV_RGB (255, 0, 0), CV_RGB (0, 255, 0), CV_RGB (255, 255, 0), CV_RGB (0, 0, 255), CV_RGB (255, 0, 255), CV_RGB (0, 255, 255)
  };

void drawHull(Mat& dst, vector<Point2i> hull, int col){
	for (int i = 0; i < hull.size(); i++){
		line(dst, hull[i], hull[(i+1)%hull.size()], color[col%6]);
	}
}

/**
 * @brief draws a vector of points in given size in cycling colors
 */
void drawApprox(Mat& dst, vector<Point> hull, int col, int size){
	for (int i = 0; i < hull.size(); i++){
		line(dst, hull[i], hull[(i+1)%hull.size()], color[col%6], size);
	}
}

/**
 * @brief draws a vector of vector of points in given size in cycling colors
 */
void drawApproxes(Mat& dst, vector<vector<Point>> app, int size = 1){
	for (int i = 0; i < app.size(); i++){
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

//! Draws a single dice on display
//! dice.eyes is written with its center at dice.p
//! if dice.eyes==0 a cross is written instead
void draw (Mat& display, const Dice& dice) {
  if (dice.eyes!=0) {
    stringstream name;
    name << dice.eyes;
    Size textSize = getTextSize (name.str(), FONT_HERSHEY_PLAIN, 8, 4, NULL);
    Point textPos = dice.p;
    textPos.x -= textSize.width/2;
    textPos.y += textSize.height/2;
    putText (display, name.str(), textPos, FONT_HERSHEY_PLAIN, 8, CV_RGB(0,255,0), 4);
  }
  else {
    drawCross (display, dice.p, 30, CV_RGB(0,255,0));
  }
}

//! Draws a list of dices on the image display
void draw (Mat& display, const vector<Dice>& dices) {
   for (int i=0; i<dices.size(); i++) draw (display, dices[i]);
}

// Makes a RGB image darker by 1/4
void makeImageDarker (Mat& display) {
   for (int y=0; y<display.rows; y++) for (int x=0; x<display.cols; x++) 
      for (int c=0; c<3; c++) {
      display.at<Vec3b>(y,x)[c] = (display.at<Vec3b>(y,x)[c] * 3)/4;
   }
}

//! Prints information about one dice found
ostream& operator<< (std::ostream& stream, const Dice& dice) {
   stream << dice.eyes << "@" << dice.p.x << "," << dice.p.y <<" ";
   return stream;
}

ostream& operator<< (std::ostream& stream, const vector<Dice>& dices) {
   for (int i=0; i<dices.size(); i++) stream << dices[i] << ends;
   stream << endl;
   return stream;
}

/**
 * @brief counts blobs as dark circles inside a white area.
 */
Dice countBlobs(SimpleBlobDetector& d, Mat& orig, RotatedRect& elem, vector<Point>& approx){
	Mat bb, M, cropped;
	//!Get bounding Rect of recognized shape
	Rect br = boundingRect(Mat(approx));
	//!Crop original image to bounding Box
    br.width -= 12;
    br.height -= 12;
    br.x += 6;
    br.y += 6;
	bb = orig(br);
    int dilatation_size = 7;
	int erosion_size = 3;

	//! This is not needed anymore
	float angle = elem.angle;
	Point2f offs(br.tl().x, br.tl().y);
	Size rect_size = elem.size;
	// thanks to http://felix.abecassis.me/2011/10/opencv-rotation-deskewing/
	if (elem.angle < -45.) {
		angle += 90.0;
		swap(rect_size.width, rect_size.height);
	}
	// get the rotation matrix
	M = getRotationMatrix2D(elem.center-offs, angle, 1.0);
	// perform the affine transformation
	Mat rotated;
	warpAffine(bb, rotated, M, bb.size(), INTER_CUBIC);
	// crop the resulting image
	getRectSubPix(rotated, rect_size, elem.center-offs, cropped);

	//Now we cropped the Dice.
// 	cropped = bb;

	//try connecting circles
	//! Apply a dilation operation to smooth holes and wipe some outside dirt
	dilate( cropped, cropped, getStructuringElement( MORPH_ELLIPSE,
            Size( 2*dilatation_size + 1, 2*dilatation_size+1 ),
            Point( dilatation_size, dilatation_size ) )
			);
    
    erode( cropped, cropped, getStructuringElement( MORPH_ELLIPSE,
            Size( 2*erosion_size + 1, 2*erosion_size+1 ),
            Point( erosion_size, erosion_size ) )
            );

	std::vector<KeyPoint> keypoints;
	d.detect(cropped, keypoints);

// 	cout << br;
	if(keypoints.size() == 0 || keypoints.size() > 6){
		cout << " Could not detect correctly at " << elem.center << "(" << keypoints.size() << ")" << endl;
	}
// 	cout << " found " << keypoints.size() << " eyes." << endl;
//  cout << "Number: " << keypoints.size() << endl;
// 	drawKeypoints(cropped, keypoints, cropped, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
// 	if(debug) showScaled("D", cropped);
// 	waitKey(1000);

	return Dice(elem.center, keypoints.size());
}


void drawWatershedImage(int count, Mat markers){
     // Generate random colors, just for the debug looks
     vector<Vec3b> colors;
     for (size_t i = 0; i < count; i++)
     {
         int b = theRNG().uniform(0, 255);
         int g = theRNG().uniform(0, 255);
         int r = theRNG().uniform(0, 255);
         colors.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
     }
     // Create the result image
     Mat dst = Mat::zeros(markers.size(), CV_8UC3);
     // Fill labeled objects with random colors
     for (int i = 0; i < markers.rows; i++)
     {
         for (int j = 0; j < markers.cols; j++)
         {
             int index = markers.at<int>(i,j);
             if (index > 0 && index <= static_cast<int>(count))
                 dst.at<Vec3b>(i,j) = colors[index-1];
             else
                 dst.at<Vec3b>(i,j) = Vec3b(0,0,0);
         }
     }
     // Visualize the final image
     showScaled("Final Result", dst);
}

/**
 * @brief Special trick to segment dices in complicated environments
 */
void cleanupFloodfill(Mat& floodfill){
	vector<vector<Point>> flood_contours;
	vector<Vec4i> hierarchy;
	double maxArea = 650, minArea = 0;
	vector<unsigned> ausschuss;
	Mat temp_image;
	floodfill.copyTo(temp_image);					//Slow but necessary
	findContours(temp_image, flood_contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
	// Find indices of contours whose area is wrong
	if ( !flood_contours.empty()) {
	    for (size_t i=0; i<flood_contours.size(); ++i) {
	        double area = contourArea(flood_contours[i]);
	        if (area < minArea || area > maxArea){
	        	cout << "ignoring elem " << i << " with area " << area << endl;
	            ausschuss.push_back(i);
	        }
	    }
	}

	//delete wrong contours from fillimage
	Mat debugShit(Size(floodfill.size()), CV_32FC3);	//debug
	Mat im_floodfill_clean;
	floodfill.copyTo(im_floodfill_clean);		//clone is expensive, but for debugging it is OK
	for(unsigned i = 0; i < ausschuss.size(); i++){
		drawContours(im_floodfill_clean, flood_contours, ausschuss[i], cv::Scalar(0), CV_FILLED, 8);
		drawContours(debugShit, flood_contours, ausschuss[i], cv::Scalar(255, 0, 0), CV_FILLED, 8);
	}

	if(debug) showScaled("im_floodfill_ignoredElems", debugShit);
	if(debug) showScaled("im_floddfill_clean", im_floodfill_clean);

	floodfill = im_floodfill_clean;
}

/**
 * @brief Segments a binimage by watershed and applies countBlobs on each found FG Element
 *
 * May contain some Sources found in http://docs.opencv.org/trunk/d2/dbd/tutorial_distance_transform.html
 */
void segmentAndRecognizeFromBinImage(Mat& binImage, vector<Dice>& dices, int& erosion_size){
    int dilatation_size = 3;
	// Floodfill from point (0, 0)
	Mat im_floodfill = binImage.clone();


	floodFill(im_floodfill, cv::Point(0,0), Scalar(255));

	// Invert floodfilled image
	Mat im_floodfill_inv;
	bitwise_not(im_floodfill, im_floodfill_inv);
	if(debug) showScaled("im_floddfill_inv vorher", im_floodfill_inv);
	bool repeat;
	Mat markers;
	vector<vector<Point> > contours;
	do{
		repeat = false;
		// Combine the two images to get the foreground.
		Mat filled = (binImage | im_floodfill_inv);
		// Perform the distance transform algorithm

		if(debug) showScaled("Filled", filled);

		Mat dist;
		distanceTransform(filled, dist, CV_DIST_L2, 3);

		dist.convertTo(dist, CV_8U);
		double max = 255;
		normalize(dist, dist, 0, max, NORM_MINMAX);

		if(debug) showScaled("Distance Transform Image", dist);
		// Threshold to obtain the peaks
		// This will be the markers for the foreground objects
		//adaptiveThreshold(dist, dist, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY, 9, -4);
		threshold(dist, dist, .84*max, 1.*max, CV_THRESH_BINARY);
		// Dilate a bit the dist image
		Mat kernel1 = Mat::ones(3, 3, CV_8UC1);
		dilate(dist, dist, kernel1);
		if(debug) showScaled("Peaks", dist);
		// Create the CV_8U version of the distance image
		// It is needed for findContours()
		Mat dist_8u;
		dist.convertTo(dist_8u, CV_8U);
		// Find total markers
		findContours(dist_8u, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
		// Create the marker image for the watershed algorithm
		markers = Mat::zeros(dist.size(), CV_32SC1);
		// Draw the foreground markers
		for (size_t i = 0; i < contours.size(); i++)
		 drawContours(markers, contours, static_cast<int>(i), Scalar::all(static_cast<int>(i)+1), -1);
		// Draw the background marker
		circle(markers, Point(5,5), 3, CV_RGB(255,255,255), -1);
		// Perform the watershed algorithm
		cvtColor(filled, filled, CV_GRAY2RGB);
		watershed(filled, markers);

		Mat mark = Mat::zeros(markers.size(), CV_8UC1);
		markers.convertTo(mark, CV_8UC1);
		bitwise_not(mark, mark);
		if(!repeat && contours.size() < 2){
			cout << "Detected less than 2 dices, retrying harder..." << endl;
			cleanupFloodfill(im_floodfill_inv);
			repeat = true;
		}
	}while(repeat);

    if(debug) drawWatershedImage(contours.size(), markers);
    
	SimpleBlobDetector::Params params;
	params.minThreshold = 0;
	params.maxThreshold = 30;
	SimpleBlobDetector detector(params);
	vector<RotatedRect> possibilities;
	cvtColor(binImage, binImage, CV_GRAY2RGB);
	if(debug) waitKey();
	for(int i = 1; i <= contours.size(); i++){
		//New Image masked with just one found element
		Mat singleElem = markers == i;
		vector<vector<Point>> singleDiceContours;
		vector<Point> singleDiceContour;
		drawApprox(binImage, singleDiceContour, i, 4);
		findContours(singleElem, singleDiceContours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

		if(singleDiceContours.size() > 1){
			cout << "single Dice was seen as multiple elems: " << singleDiceContours.size() << endl;
		}
		singleDiceContour = singleDiceContours[0];
		RotatedRect elem = minAreaRect(singleDiceContour);

		possibilities.push_back(elem);
		dices.push_back(countBlobs(detector, binImage, elem, singleDiceContour));
	}
	drawRects(binImage, possibilities);
	if(debug) showScaled("BinImage", binImage);
	cout << "Found " << possibilities.size() << " dices." << endl;

}

/********** BELOW HERE TODOs ******************************/

void segmentDices(Mat& image, Mat& display, vector<Dice>& dices){
	Mat yellow_bin;
    Mat blue_bin;
    Mat white_bin;
    
    Mat hsv;
    cvtColor(image, hsv, CV_BGR2HSV);
    
	unsigned char RETR = CV_RETR_FLOODFILL, CHAIN = CV_CHAIN_APPROX_TC89_KCOS ;
	int erosion_size = 6;

	Rect rect(0, 0, image.size().width * 0.77, image.size().height * 0.9);

//  inRange(hsv, Scalar(yl, bl, wl), Scalar(yh, bh, wh), blue_bin);

	inRange(hsv, Scalar(0, 0, 175), Scalar(180, 75, 255), white_bin);    //WHITE
	Mat white_crop = white_bin(rect);                                    //WHITE_CROPPED
	inRange(hsv, Scalar(0, 75, 160), Scalar(50, 255, 255), yellow_bin);  //YELLOW
	Mat yellow_crop = yellow_bin(rect);                                  //YELLOW_CROPPED
	inRange(hsv, Scalar(100, 60, 90), Scalar(120, 255, 185), blue_bin);  //BLUE
	Mat blue_crop = blue_bin(rect);                                      //BLUE_CROPPED

	segmentAndRecognizeFromBinImage(blue_crop, dices, erosion_size);
	segmentAndRecognizeFromBinImage(white_crop, dices, erosion_size);
	segmentAndRecognizeFromBinImage(yellow_crop, dices, erosion_size);
	//draw (blue_crop, dices);
	if(debug) showScaled("test", blue_crop);
}

//! Recognizes all dices in images and returns them in dices
//! display is passed, just in case you want to show something
//! for debugging reasons
void findDices (Mat& image, Mat& display, vector<Dice>& dices) {
	segmentDices(image, display, dices);
}

void addToStatistics (vector<int>& statistics, const vector<Dice>& dices)
{
    for (int i=0; i < dices.size(); i++) {
        statistics[dices[i].eyes]++;
    }
}

//! Returns whether the distribution of eyes in statistics in compatible
//! with the hypothesis of a uniform distribution [1..6]
//! The function executes a chi-square-distribution test with
//! a significance level of 90%.
bool passed (const vector<int>& statistics) {

    int n = 0; 
    float xemp = 0;
    for (int i=1; i <= 6; i++) {
        n = n+statistics[i];
    }

    int ne = n/6;

    for (int i=1; i <= 6; i++) {
        xemp = xemp + (pow((statistics[i]-ne), 2) / ne);
        //std::cout << "DEBUG: xemp for " << i << " Augen = " << xemp << std::endl;
    }
    if (xemp < 1.61) { // 1,61 Ist ist die Grenze bei 5 Freiheitsgraden und einem 90% Testniveau
        return true;
    } else {
        return false;
    }
}


/***************************************************************************/

int main( int argc, char** argv )
{
    if (argc==1) {
       cerr << "Usage" << endl;
       cerr << "dices [image] ... [image]" <<endl;
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


       vector<Dice> dices;
       findDices (src, display, dices);
       cout << "Found the following dices: "<< endl << dices << endl;
       addToStatistics (statistics, dices);
       draw (display, dices);

       showScaled (windowname, display);
       if (waitKey()==27) break;
       imageCtr++;
    } 
    cout << endl << endl;
    cout << "Statistics: " ;
    for (int i=1; i<=6; i++) cout << statistics[i] << " ";
    cout << endl;
    if (passed(statistics)) {
       cout << "The distribution of dice eyes is OKAY." << endl;
       return 0;
    }
    else {
       cout << "The distribution of dice eyes is WRONG (F=90%)." << endl;
       return 1;
    }
}
