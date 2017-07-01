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
}

void hardSegmentation(cv::Mat &input, std::vector<cv::Point2f> &output){
	output.push_back(Point(0,0));
	output.push_back(Point(input.size().width,0));
	output.push_back(Point(input.size()));
	output.push_back(Point(0,input.size().height));

	Mat canny_output;
   vector<vector<Point> > contours;
   vector<Vec4i> hierarchy;
   Mat dummy;
   /// Detect edges using canny

   double perfectThresholdBelieveMe = threshold(input, dummy, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

   Canny(input, canny_output, perfectThresholdBelieveMe/2, perfectThresholdBelieveMe);

   showScaled("canny output", canny_output);

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
    
    
    
    return res;
}

static CustomOCR customOcr;

/**
 * @return true, if plate is one of the known plates
 */
int megaPlateRecognisificationessing(cv::Mat &input){
	if(type2str(input.type()) != string("8UC1")){
		cout << "Converting input image to grey." << endl;
		cvtColor(input, input, COLOR_RGB2GRAY);
	}
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
