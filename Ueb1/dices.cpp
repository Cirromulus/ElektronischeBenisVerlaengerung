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

using namespace std;
using namespace cv;

// The images are very large, hence for displaying they are scaled down to
// the size below.
int windowWidth = 1200, windowHeight = 600;



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

void drawApprox(Mat& dst, vector<Point> hull, int col, int size){
	for (int i = 0; i < hull.size(); i++){
		line(dst, hull[i], hull[(i+1)%hull.size()], color[col%6], size);
	}
}

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


Dice countBlobs(SimpleBlobDetector& d, Mat& orig, RotatedRect& elem, vector<Point>& approx){
	Mat bb, M, cropped, rotated;

	Rect br = boundingRect(Mat(approx));
	cout << br << endl;
	bb = orig(br);

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
	warpAffine(bb, rotated, M, bb.size(), INTER_CUBIC);
	// crop the resulting image
	getRectSubPix(rotated, rect_size, elem.center-offs, cropped);

	//Now we cropped the Dice.
	//showScaled("F", bb);
	//showScaled("R", rotated);

	std::vector<KeyPoint> keypoints;
	d.detect(cropped, keypoints);

	//cout << "Number: " << keypoints.size() << endl;
	drawKeypoints(cropped, keypoints, cropped, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
	showScaled("D", cropped);
	waitKey(100);

	return Dice(elem.center, keypoints.size());
}


/********** BELOW HERE TODOs ******************************/

void idea1(Mat& image, Mat& display, vector<Dice>& dices){
	Mat yellow_bin;
	Mat canny_output;
	unsigned char key = 0;
	unsigned char inc = 10;
	unsigned char lr = 0x09, lg = 0xA0, lb = 0x6F,
			hr = 0x6E, hg = 0xFF, hb = 0xFF;
	unsigned char RETR = CV_RETR_FLOODFILL, CHAIN = CV_CHAIN_APPROX_TC89_KCOS ;
	int erosion_size = 2;
	//yellow 09 A0 6F, 6E FF FF
	do{
		dices.clear();	//FIXME only for debug
		switch(key){
		case '\n':
			continue;
		case '1':
			inc -= 5;
			break;
		case '2':
			inc += 5;
			break;
		case 'w':
			lr += inc;
			break;
		case 's':
			lr -= inc;
			break;
		case 'e':
			lg += inc;
			break;
		case 'd':
			lg -= inc;
			break;
		case 'r':
			lb += inc;
			break;
		case 'f':
			lb -= inc;
			break;
		case 'u':
			hr += inc;
			break;
		case 'j':
			hr -= inc;
			break;
		case 'i':
			hg += inc;
			break;
		case 'k':
			hg -= inc;
			break;
		case 'o':
			hb += inc;
			break;
		case 'l':
			hb -= inc;
			break;
		case 'x':
			erosion_size--;
			break;
		case 'c':
			erosion_size++;
			break;
		default:
			printf("Fucktard\n");

		}
		printf("s: %d, l: %02X %02X %02X, h: %02X %02X %02X, erosion: %d\n", inc, lr, lg, lb, hr, hg, hb, erosion_size);
		inRange(image, Scalar(lr, lg, lb), Scalar(hr, hg, hb), yellow_bin);

	    // Floodfill from point (0, 0)
	    Mat im_floodfill = yellow_bin.clone();
	    floodFill(im_floodfill, cv::Point(0,0), Scalar(255));

	    // Invert floodfilled image
	    Mat im_floodfill_inv;
	    bitwise_not(im_floodfill, im_floodfill_inv);

	    // Combine the two images to get the foreground.
	    Mat im_out = (yellow_bin | im_floodfill_inv);
	    // Perform the distance transform algorithm
		//http://docs.opencv.org/trunk/d2/dbd/tutorial_distance_transform.html

	    //showScaled("Filled", im_out);

	    Mat dist;
	    distanceTransform(im_out, dist, CV_DIST_L2, 3);

	    normalize(dist, dist, 0, 1., NORM_MINMAX);
	    //showScaled("Distance Transform Image", dist);

		// Threshold to obtain the peaks
		// This will be the markers for the foreground objects
		threshold(dist, dist, .5, 1., CV_THRESH_BINARY);
		// Dilate a bit the dist image
		Mat kernel1 = Mat::ones(3, 3, CV_8UC1);
		dilate(dist, dist, kernel1);
		//showScaled("Peaks", dist);
		// Create the CV_8U version of the distance image
		// It is needed for findContours()
		Mat dist_8u;
		dist.convertTo(dist_8u, CV_8U);
		// Find total markers
		vector<vector<Point> > contours;
		findContours(dist_8u, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
		// Create the marker image for the watershed algorithm
		Mat markers = Mat::zeros(dist.size(), CV_32SC1);
		// Draw the foreground markers
		for (size_t i = 0; i < contours.size(); i++)
		 drawContours(markers, contours, static_cast<int>(i), Scalar::all(static_cast<int>(i)+1), -1);
		// Draw the background marker
		circle(markers, Point(5,5), 3, CV_RGB(255,255,255), -1);
		// Perform the watershed algorithm
		watershed(image, markers);
		Mat mark = Mat::zeros(markers.size(), CV_8UC1);
		markers.convertTo(mark, CV_8UC1);
		bitwise_not(mark, mark);
		showScaled("Markers_v2", mark); // uncomment this if you want to see how the mark
	                                   // image looks like at that point

	    // Generate random colors
		vector<Vec3b> colors;
		for (size_t i = 0; i < contours.size(); i++)
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
				if (index > 0 && index <= static_cast<int>(contours.size()))
					dst.at<Vec3b>(i,j) = colors[index-1];
				else
					dst.at<Vec3b>(i,j) = Vec3b(0,0,0);
			}
		}
		// Visualize the final image
		showScaled("Final Result", dst);


		/*
		Mat element = getStructuringElement( MORPH_CROSS,
										   Size( 2*erosion_size + 1, 2*erosion_size+1 ),
										   Point( erosion_size, erosion_size ) );

		Mat erod;
		yellow_bin.copyTo(erod);
		/// Apply the erosion operation
		erode( yellow_bin, erod, element );
		//Possibility: http://answers.opencv.org/question/46525/rotation-detection-based-on-template-matching/
		//docs.opencv.org/3.1.0/d3/db4/tutorial_py_watershed.html
		//http://cmm.ensmp.fr/~beucher/wtshed.html
		vector<vector<Point> > contours;
		vector<Vec4i> hierarchy;
		vector<Point> approx;
		Canny(erod, canny_output, 100, 255, 3);
		findContours( canny_output, contours, hierarchy, RETR+1, CHAIN+1, Point(0, 0) );
*/

		/*SimpleBlobDetector::Params params;
		params.minThreshold = 10;
		params.maxThreshold = 250;
		SimpleBlobDetector detector(params);
		vector<RotatedRect> possibilities;
		for(auto contour : contours){
			vector<Point> approx;
			approxPolyDP(Mat(contour), approx, 2, true);
			RotatedRect elem = minAreaRect(approx);

			if(elem.size.width < 80 || elem.size.height < 80){
			   //cout << "Kantenl'nge too small (" << elem.size.width << "," << elem.size.height << ")" << endl;
			   continue;
			}

			if(elem.size.width > 140 || elem.size.height > 140){
			   //cout << "Kantenl'nge too big (" << elem.size.width << "," << elem.size.height << ")" << endl;
			   continue;
			}

			//todo: further checking

			bool nah = false;
			for(RotatedRect possibility : possibilities){
				if(norm(possibility.center-elem.center) < 40){
					nah = true;
					break;
				}
			}
			if(nah){
				cout << "too close to another found dice" << endl;
				continue;
			}

			//Nachbearbeitung
			elem.center -= Point2f(erosion_size,erosion_size);
			elem.size += Size2f(2*erosion_size,2*erosion_size);

			possibilities.push_back(elem);
			//dices.push_back(countBlobs(detector, yellow_bin, elem, approx));
		}

		cvtColor(yellow_bin, yellow_bin, CV_GRAY2RGB);
		//drawApproxes(display, contours, 4);
		drawApproxes(erod, contours, 6);
		//drawRects(display, possibilities);
		drawRects(erod, possibilities);
		showScaled("testnme", erod);

		cout << "Found " << possibilities.size() << " dices." << endl;*/
		waitKey();
	}while(false);//(key = getchar()) != 'q');

}

void idea2(Mat& image, Mat& display, vector<Dice>& dices){
	Mat yellow_bin;
	Mat canny_output;
	unsigned char lr = 0x09, lg = 0x95, lb = 0x6F,
			hr = 0x6E, hg = 0xFF, hb = 0xFF;
	inRange(image, Scalar(lr, lg, lb), Scalar(hr, hg, hb), yellow_bin);

	Mat shure_bg;
	dilate(yellow_bin, shure_bg, Mat(),Point(), 2);
	Mat dist_transform;
	distanceTransform(yellow_bin, dist_transform, CV_DIST_L2, 5);
	Mat shure_fg;
	double th = threshold(dist_transform, shure_fg, 127, 255, 0);
	shure_fg.convertTo(shure_fg, CV_8U);
	cout << type2str(shure_bg.type()) << " " << type2str(shure_fg.type()) << endl;
	Mat unknown = shure_bg-shure_fg;

	showScaled("Shure fg", shure_fg);
	showScaled("Shure bg", shure_bg);
	showScaled("unnoknwd", unknown);
}
//! Recognizes all dices in images and returns them in dices
//! display is passed, just in case you want to show something
//! for debugging reasons
void findDices (Mat& image, Mat& display, vector<Dice>& dices) {
   // TODO: Implement
	idea1(image, display, dices);
}

//! Adds a list of recognized dices to statistics
//! statistics[i] shows how often overall Dice.eyes was i
void addToStatistics (vector<int>& statistics, const vector<Dice>& dices)
{
   //TODO: Implement
}

//! Returns whether the distribution of eyes in statistics in compatible
//! with the hypothesis of a uniform distribution [1..6]
//! The function executes a chi-square-distribution test with
//! a significance level of 90%.
bool passed (const vector<int>& statistics) {
   //TODO: Implement
   return true;
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
