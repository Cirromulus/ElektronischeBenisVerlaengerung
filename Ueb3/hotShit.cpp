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
#include "opencv2/imgcodecs.hpp"
#include "opencv2/core/utility.hpp"

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <climits>

using namespace std;
using namespace  cv;




void tightPreprocessing(cv::Mat &img, bool live){
    cvtColor(img, img, COLOR_RGB2GRAY);
    if (debug) imshow("Greyscale (input)", img);

	// Set threshold and maxValue
   // double thresh = 80;

	//static img params
	uchar in_min=140;
	uchar in_max=255;
	double gamma=1.2;
	uchar out_min=0;
	uchar out_max=255;

    if(live){
		//Live optimized params
		in_min= 150;
		in_max= 255;
		gamma = 0.5;
		out_min=0;
		out_max=255;
    }

    double pixel = 0;

    //Note: color correction algorithm inspired by [https://pippin.gimp.org/image-processing/chap_point.html]
    for( int y = 0; y < img.rows; y++ )
       { for( int x = 0; x < img.cols; x++ )
            {
    	   	   	pixel = img.at<uchar>(y,x);
    	   	   	  // normalize
				pixel = (pixel-in_min) / (in_max-in_min);
				  // transform gamma
				pixel= (pixel > 0) ? pow(pixel,gamma) : 0;
				  //rescale range and round correctly
				pixel = floor((pixel * (out_max-out_min) + out_min)+0.5);
    	   	    img.at<uchar>(y,x) = saturate_cast <uchar> (pixel);
              }
       }
    //equalizeHist( new_image, new_image );

   if (debug){
	   imshow( "Preprocessed image", img);
   }
}

void hardSegmentation(cv::Mat input, std::vector<cv::Point2f> &output){
// 	//Testing code
//  	 if(false){
// 		float rein = 1;
// 		output.push_back(Point2f(rein,rein));
// 		output.push_back(Point2f(input.size().width-rein,rein));
// 		output.push_back(Point2f(input.size()) - Point2f(rein, rein));
// 		output.push_back(Point2f(rein,input.size().height - rein));
// 	}else{
// 		if(true){
// 			//Img 1969
// 			output.push_back(Point2f(240,213));
// 			output.push_back(Point2f(680,196));
// 			output.push_back(Point2f(676,285));
// 			output.push_back(Point2f(246,315));
// 		}else{
// 			//Img 1962
// 			output.push_back(Point2f(195,216));
// 			output.push_back(Point2f(300,213));
// 			output.push_back(Point2f(300,309));
// 			output.push_back(Point2f(200,314));
// 		}
// // 		float weg = 10;
// // 		output[0] += Point2f(-weg, -weg);
// // 		output[1] += Point2f( weg, -weg);
// // 		output[2] += Point2f( weg,  weg);
// // 		output[3] += Point2f(-weg,  weg);
//     return;
// 	}
	

	int edgeThresh_lower = 100;
	RNG rng(12345);
	Mat edge, cedge, im_flood, im_contours;
	vector<vector<Point>> foundContours;
	vector<vector<Point>> foundConvexHulls;
	vector<vector<Point>> filteredContours;
	vector<Vec4i> hierarchy;

	//blurring input and converting to grey

	blur(input, edge, Size(3,3));
	cvtColor(input, input, COLOR_GRAY2RGB);

	// Run the edge detector on blurred input
	Canny(edge, edge, edgeThresh_lower, edgeThresh_lower*5, 3);

	if(debug){
		cedge = Scalar::all(0);
		input.copyTo(cedge, edge);
		imshow("Edge map", cedge);
	}

	//find contours in canny output
	findContours(edge, foundContours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_TC89_L1);
	foundConvexHulls.resize(foundContours.size());
	for(unsigned int i = 0; i < foundContours.size(); i++){
		convexHull(foundContours[i], foundConvexHulls[i]);
	}

	if(foundConvexHulls.size()>0){

		//storage for largest contour
		int area = 0;
		int max_area=0;
		int threshold_approximation=3;
		vector<vector<Point>> largest_contour;
		largest_contour.resize(1);

		im_contours=input.clone();

		filteredContours.resize(foundConvexHulls.size());

		//srtorage for Points of final contour with corresponding distance to origin
		vector<accDistanceAndPoint> sortedPoints;

		//get largest contour by successive comparison of area sizes
		for( size_t k = 0; k < foundConvexHulls.size(); k++ ){
			approxPolyDP(Mat(foundConvexHulls[k]), filteredContours[k],threshold_approximation, true);
			area = contourArea(filteredContours[k]);

			if(area>=max_area){
				max_area =  area;
				if (debug) cout << "max area: " << max_area << endl;
				largest_contour[0]=filteredContours[k];
			}
		}

		if(debug){  //show points of found contour after first approximation
			cout << "largest contour: " << largest_contour[0] << endl;
			for(Point pt : largest_contour[0]){
				drawCross(im_contours,pt,4,Scalar(0,255,0));
			}
		}

		//simplifying contour until polygon has only 4 corners
		while(largest_contour[0].size()>4){
			approxPolyDP(Mat(largest_contour[0]), largest_contour[0],threshold_approximation, true);
			threshold_approximation++;
		}

		//storing points of final contour and the accumulated distance to 0 (x+y) in sortedPoints
		for(Point pt : largest_contour[0]){
			accDistanceAndPoint elem;
			elem.accDist=pt.x+pt.y;
			elem.pt=pt;
			sortedPoints.push_back(elem);
		}

		//sort points by accumulated distance -> order will be: top_left, bottom_left, top_right, bottom_right
		sort(sortedPoints.begin(), sortedPoints.end());

		//reposition element as order should be: top_left, top_right, bottom_right, bottom_left
		sortedPoints.push_back(sortedPoints[1]);
		sortedPoints.erase(sortedPoints.begin()+1);

		//output points
		for(accDistanceAndPoint elem : sortedPoints){
			output.push_back(cv::Point2f(elem.pt));
		}


		//Additional output for debug mode
		if(debug){

		    namedWindow("Edge map", 1);
		    namedWindow("source",1);
		    namedWindow("contours",1);

			//show points of found contour after simplification to only 4 points
			cout << "largest contour after approximation: " << largest_contour[0] << endl;

			for( uint i = 0; i<filteredContours.size(); i++ ){
				Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
				drawContours(im_contours, filteredContours, i, color, 2, 8, hierarchy, 0, Point() );
//				cout << "area of contour Nr." << i << ": " << contourArea(filteredContours[i]) << endl;
//				imshow("contours",im_contours);
//				waitKey(0);
			}
			for(uint i=0; i<filteredContours.size(); i++){
						cout << "i=" << i << " ";
						for(uint j=0; j<filteredContours[i].size(); j++){
							cout << ":" << filteredContours[i][j];
						}
					cout << endl;
			}

			drawContours(im_contours,largest_contour,-1, Scalar(255,0,0),2,8,hierarchy, 0, Point() );
			for(Point pt : largest_contour[0]){
				drawCross(im_contours,pt,4,Scalar(0,0,255));
			}

			cout << "Corner points of plate region: " << output << endl;


		    imshow("source", input);
			imshow("contours",im_contours);

		}

	}
}


//Also would crop image
cv::Mat phatPerspectiveNormalizer(cv::Mat &input, std::vector<cv::Point2f> &outline){
	if(outline.size() != 4){
		if(debug) cout << "Outline has " << outline.size() << " edges. Skipping Perspective Warp." << endl;
		return input;
	}
    string croppedStr = "cropped";
    string transformedStr = "transformed";
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
   
    int plateWidth, plateHeight;
    //determinine plate format
    if(width / height < 1.8)    //two-rowed plate
    {
        if(height >= 200)
        {
            plateHeight = 200;
            plateWidth= 340;
        }
        else
        {
            plateHeight = height;
            plateWidth = height * 1.7; // 1.7 = 340/200
        }
    }
    else //single-rowed plate
    {
        if(height >= 110)
        {
            plateHeight = 110;
            plateWidth= 520;
        }
        else
        {
            plateHeight = height;
            plateWidth = height * 4.73; //4.73 = 520mm / 110mm
        }
    }
    
    
    
    std::vector<cv::Point2f> destPoints = {cv::Point2f(0, 0), cv::Point2f(plateWidth, 0), cv::Point2f(plateWidth, plateHeight), cv::Point2f(0, plateHeight)};
    
    std::vector<cv::Point2f> srcPoints;

    for(unsigned int i = 0; i < outline.size(); i++)
    {
        srcPoints.push_back(outline[i] - Point2f(x_min, y_min));
    }
    
    if(debug){
		cout << "scrPoints: " << srcPoints << endl;
		cout << "destPoints: " << destPoints << endl;
	}

    Mat trans = cv::getPerspectiveTransform(srcPoints, destPoints);
    Mat res(plateHeight, plateWidth, croppedImg.type());
    
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
		if(debug) cout << "Converting input image to gray, because someone forgot that" << endl;
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
