// Implement a solution to localize and identify the objects in the image
// from the set of possible objects (plate, fork, knife, spoon). The position
// where to add code is marked by "TODO".
// Consider
// http://docs.opencv.org/3.1.0/d3/dc0/group__imgproc__shape.html

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
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
   cout << "Scaled " << display.cols << "*" << display.rows << " to " << wx << "*" << wy << endl;
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

void drawApprox(Mat& dst, vector<Point> hull, int col){
	for (int i = 0; i < hull.size(); i++){
		line(dst, hull[i], hull[(i+1)%hull.size()], color[col%6]);
	}
}

void drawApproxes(Mat& dst, vector<vector<Point>> app){
	for (int i = 0; i < app.size(); i++){
		drawApprox(dst, app[i], i);
	}
}

void drawRect(Mat& dst, RotatedRect& rec, int i) {
  Scalar myColor = color[i%6];
  Point2f vertices[4];
  rec.points(vertices);
  for (int i = 0; i < 4; i++)
    line(dst, vertices[i], vertices[(i+1)%4], myColor, 2);
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



/********** BELOW HERE TODOs ******************************/

//! Recognizes all dices in images and returns them in dices
//! display is passed, just in case you want to show something
//! for debugging reasons
void findDices (Mat& image, Mat& display, vector<Dice>& dices) {
   // TODO: Implement
	Mat yellow_bin;
	Mat canny_output;
	unsigned char key = 0;
	unsigned char inc = 10;
	unsigned char lr = 0x09, lg = 0xA5, lb = 0x6F,
			hr = 0x6E, hg = 0xFF, hb = 0xFF;
	//l: 13 A5 BF, h: 50 D0 EB
	//l: 09 A5 6F, h: 6E F8 FF
	do{
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
		default:
			printf("Fucktard\n");

		}
		printf("s: %d, l: %02X %02X %02X, h: %02X %02X %02X\n", inc, lr, lg, lb, hr, hg, hb);
		inRange(image, Scalar(lr, lg, lb), Scalar(hr, hg, hb), yellow_bin);

		//Possibility: http://answers.opencv.org/question/46525/rotation-detection-based-on-template-matching/
		vector<vector<Point> > contours;
		vector<Vec4i> hierarchy;
		vector<Point> approx;
		Canny(yellow_bin, canny_output, 100, 255, 3);
		findContours( canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_TC89_L1, Point(0, 0) );

		vector<RotatedRect> possibilities;
		for(auto contour : contours){
			vector<Point> approx;
			approxPolyDP(Mat(contour), approx, 2, true);
			RotatedRect elem = minAreaRect(approx);

			if(elem.size.width < 80 || elem.size.height < 80){
			   //cout << "Kantenl'nge too small (" << elem.size.width << "," << elem.size.height << ")" << endl;
			   continue;
			}

			//todo: further checking

			possibilities.push_back(elem);
		}

		cvtColor(yellow_bin, yellow_bin, CV_GRAY2RGB);
		drawApproxes(display, contours);
		drawApproxes(yellow_bin, contours);
		drawRects(display, possibilities);
		drawRects(yellow_bin, possibilities);
		showScaled("testnme", yellow_bin);
		waitKey(500);
	}while((key = getchar()) != 'q');

   dices.clear();
   dices.push_back (Dice(Point(image.cols/2, image.rows/2), 5));
   dices.push_back (Dice(Point(100,100),0));
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
