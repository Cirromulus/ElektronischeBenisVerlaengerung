/*
 * hotShit.cpp
 *
 *  Created on: Jun 20, 2017
 *      Author: rooty
 */

#include "hotShit.hpp"
#include "helpers.hpp"
#include "ocrBackend.hpp"
#include "knownPlates.hpp"

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <climits>

using namespace std;
using namespace  cv;

void tightPreprocessing(cv::Mat &img){
    cvtColor(img, img, COLOR_RGB2GRAY);
    //This is python code and thus cant be used in a c++ compiler
    //threshold(img,127,255,cv2.THRESH_TOZERO);

    //This normalizes Values to 1..0, but our greyscale image is from 0..2^8 (or higher)
    //normalize(img, img, 1, 0,NORM_MINMAX);
}

void hardSegmentation(cv::Mat &input, std::vector<cv::Point2f> &output){

	if(false){
		float rein = 1;
		output.push_back(Point2f(rein,rein));
		output.push_back(Point2f(input.size().width-rein,rein));
		output.push_back(Point2f(input.size()) - Point2f(rein, rein));
		output.push_back(Point2f(rein,input.size().height - rein));
	}else{
		if(true){
			//Img 1969
			output.push_back(Point2f(240,213));
			output.push_back(Point2f(680,196));
			output.push_back(Point2f(676,285));
			output.push_back(Point2f(246,315));
		}else{
			//Img 1962
			output.push_back(Point2f(195,216));
			output.push_back(Point2f(707,213));
			output.push_back(Point2f(701,309));
			output.push_back(Point2f(200,314));
		}
		float weg = 10;
		output[0] += Point2f(-weg, -weg);
		output[1] += Point2f( weg, -weg);
		output[2] += Point2f( weg,  weg);
		output[3] += Point2f(-weg,  weg);
	}

	Mat canny_output;
   vector<vector<Point> > contours;
   vector<Vec4i> hierarchy;
   Mat dummy;
   /// Detect edges using canny

   double perfectThresholdBelieveMe = threshold(input, dummy, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

   Canny(input, canny_output, perfectThresholdBelieveMe/2, perfectThresholdBelieveMe);

   //if(debug) showScaled("canny output", canny_output);

   /// Find contours
   //findContours( canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_TC89_L1, Point(0, 0) );

}

//Also would crop image
cv::Mat phatPerspectiveNormalizer(cv::Mat &input, std::vector<cv::Point2f> &outline){
    string croppedStr = "cropped";
    string transformedStr = "transormed";
    if(debug) {
        namedWindow( croppedStr, WINDOW_AUTOSIZE );
        namedWindow( transformedStr, WINDOW_AUTOSIZE );
    }
    float x_min, x_max, y_min, y_max;
    x_max = y_max = std::numeric_limits<float>::min();;
    x_min = y_min = std::numeric_limits<float>::max();;
    for(cv::Point pt : outline) {
        if(pt.x < x_min) x_min = pt.x;
        if(pt.x > x_max) x_max = pt.x;
        if(pt.y < y_min) y_min = pt.y;
        if(pt.y > y_max) y_max = pt.y;
    }
    
    float width = x_max-x_min, height = y_max-y_min;
    cv::Rect cropRect(x_min, y_min, width, height);
    //cv::Rect cropRect(0, 0, input.size().width, input.size().height);
    cv::Mat croppedImg = input(cropRect);
    if(debug) showScaled(croppedStr, croppedImg);
   
    std::vector<cv::Point2f> destPoints = {cv::Point2f(0, 0), cv::Point2f(width, 0), cv::Point2f(width, height), cv::Point2f(0, height)};
    
    std::vector<cv::Point2f> srcPoints;
    for(unsigned int i = 0; i < outline.size(); i++)
    {
        srcPoints.push_back(outline[i] - Point2f(x_min, y_min));
    }
    
    Mat trans = cv::getPerspectiveTransform(srcPoints, destPoints);
    Mat res = input(cropRect);
    
    cv::warpPerspective(croppedImg, res, trans, res.size());
    
    if(debug) showScaled(transformedStr, res);
    
    return res.clone();
}

/**
 * @return true, if plate is one of the known plates
 */
int megaPlateRecognisificationessing(cv::Mat &input){
	static CustomOCR customOcr;
	//static LexiconOCR customOcr;

	if(type2str(input.type()) != string("8UC1")){
		cout << "Converting input image to gray." << endl;
		cvtColor(input, input, COLOR_RGB2GRAY);
	}
	if(debug) cout << "Type: " << type2str(input.type()) << endl;
	if(debug) cout << "Size: " << input.size() << endl;
	std::vector<std::string> texts = customOcr.ocr(input);
	double t_r = (double)getTickCount();
	int found = -1;
	for(auto text : texts){
		for(unsigned int i = 0; i < numberOfKnownPlates; i++){
			char *plate = strdup(knownPlates[i]);
			if(debug){
				cout << "Scanning for " << string(plate) << "... ";
			}
			char *subelem = strtok(plate, ":");
			bool valid = true;
			while(subelem) {
				if(!strstr(text.c_str(), subelem)){
					//Not found
					if(debug) cout << "'" << string (subelem) << "' not found" << endl;
					valid = false;
					break;
				}
				subelem = strtok(NULL, ":");
			}
			if(valid){
				if(debug) cout << "found." << endl;
				found = i;
			}
		}
		if(found >= 0){
			break;
		}
	}
	if(debug) cout << "TIME_PLATE_SCANNING = " << ((double)getTickCount() - t_r)*1000/getTickFrequency() << endl;
	return found;
}
