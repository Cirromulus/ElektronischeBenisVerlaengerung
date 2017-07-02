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




void tightPreprocessing(cv::Mat &img){
    cvtColor(img, img, COLOR_RGB2GRAY);
    imshow("greyscale", img);

    Mat new_image =  img.clone();

	// Set threshold and maxValue
   // double thresh = 80;


    uchar in_min=180;
    uchar in_max=255;
    uchar gamma=1;
    uchar out_min=0;
    uchar out_max=255;

    double cutoff = 120;
    double pixel = 0;

//    // Binary Threshold
//    threshold(img,img, thresh, maxValue, THRESH_TOZERO);

    //Note: color correction algorithm inspired by [https://pippin.gimp.org/image-processing/chap_point.html]
    for( int y = 0; y < img.rows; y++ )
       { for( int x = 0; x < img.cols; x++ )
            {
    	   	   	pixel = (img.at<uchar>(y,x) > cutoff) ? img.at<uchar>(y,x) : 0;
    	   	   	  // normalize
				pixel = (pixel-in_min) / (in_max-in_min);
				  // transform gamma
				pixel= pow(pixel,gamma);
				  //rescale range
				pixel = pixel * (out_max-out_min) + out_min;

    	   	    new_image.at<uchar>(y,x) = saturate_cast <uchar> (pixel);
              }
       }
    //equalizeHist( new_image, new_image );

   if (debug){
	   imshow( "Preprocessed image", new_image);
	   waitKey(0);}
}

void hardSegmentation(cv::Mat &input, std::vector<cv::Point2f> &output){

	//Struct for storing Point and accumulated distance to origin
	struct accDistanceAndPoint{
		Point pt;
		int accDist;

		//overwriting operator to make sorting of accDistanceAndPoint possible by comparing accDist
		inline bool operator<(const accDistanceAndPoint& a) const{
			return  accDist < a.accDist;
		}
	//	inline bool operator!=(accDistanceAndPoint left, accDistanceAndPoint right){
	//		return left.accDist != right.accDist;
	//	}


	//		//returns true if distance 1 is smaller than distance 2
	//		bool compareDistances(accDistanceAndPoint obj1, accDistanceAndPoint obj2){
	//			int dist1 = obj1.accDist;
	//			int dist2 = obj2.accDist;
	//			return (dist1<dist2);
	//
	//		}

	};


	int edgeThresh_lower = 100;

	RNG rng(12345);

	Mat edge, cedge, im_flood, im_contours;

	vector<vector<Point>> foundContours;
	vector<vector<Point>> filteredContours;

	vector<Vec4i> hierarchy;

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
	if(foundContours.size()>0){
		input.copyTo(im_contours, input);

		filteredContours.resize(foundContours.size());

		//storage for largest contour
		int area = 0;
		int max_area=0;
		int threshold_approximation=3;
		vector<vector<Point>> largest_contour;
		largest_contour.resize(1);

		//srtorage for Points of final contour with corresponding distance to origin
		vector<accDistanceAndPoint> sortedPoints;

		//get largest contour by successive comparison of area sizes
		for( size_t k = 0; k < foundContours.size(); k++ ){
			approxPolyDP(Mat(foundContours[k]), filteredContours[k],threshold_approximation, true);
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

		//sort points by accumlated distance -> order will be: top_left, bottom_left, top_right, bottom_right
		sort(sortedPoints.begin(), sortedPoints.end());

		//switch positions of top_right and bottom_right for coherent representation of contour
		accDistanceAndPoint bottom_right_corner=sortedPoints[3];
		sortedPoints[3]=sortedPoints[2];
		sortedPoints[2]=bottom_right_corner;


		for(accDistanceAndPoint elem : sortedPoints){
			output.push_back(cv::Point2f(elem.pt));
		}

		if(debug){	//show points of found contour after simplificatition to only 4 points
			cout << "largest contour after approximation: " << largest_contour[0] << endl;

			for( uint i = 0; i<filteredContours.size(); i++ ){
				Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
				drawContours(im_contours, filteredContours, i, color, 2, 8, hierarchy, 0, Point() );
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

		    namedWindow("Edge map", 1);
		    namedWindow("source",1);
		    namedWindow("contours",1);

		    imshow("source", input);
			imshow("contours",im_contours);

		}

	}
//
//
//	    // create a toolbar
//	    createTrackbar("Binary lower threshold", "Edge map", &binThresh, 255, onTrackbar);
//	    createTrackbar("Canny lower threshold", "Edge map", &edgeThresh_lower, 255, onTrackbar);
//	    createTrackbar("Canny uppper threshold", "Edge map", &edgeThresh_upper, 255, onTrackbar);
//
//	    createTrackbar("floodfill lower threshold", "floodfill", &floodThresh_lower, 255, onTrackbarTwo);
//	    createTrackbar("floodfill upper threshold", "floodfill", &floodThresh_upper, 255, onTrackbarTwo);


}


//Also would crop image
cv::Mat phatPerspectiveNormalizer(cv::Mat &input, std::vector<cv::Point2f> &outline){
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
    cv::Mat croppedImg = input(cropRect);
    if(debug) showScaled(croppedStr, croppedImg);
   
    std::vector<cv::Point2f> destPoints = {cv::Point2f(0, 0), cv::Point2f(width, 0), cv::Point2f(width, height), cv::Point2f(0, height)};
    
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
