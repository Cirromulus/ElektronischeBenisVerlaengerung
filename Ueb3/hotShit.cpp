/*
 * hotShit.cpp
 *
 *  Created on: Jun 20, 2017
 *      Author: rooty
 */

#include "hotShit.hpp"
#include "helpers.hpp"

using namespace std;
using namespace  cv;

void tightPreprocessing(cv::Mat &img){
	(void) img;
}

void hardSegmentation(cv::Mat &input, std::vector<cv::Point> &output){
	output.push_back(Point(0,0));
	output.push_back(Point(input.size().width,0));
	output.push_back(Point(input.size()));
	output.push_back(Point(0,input.size().height));
}

void phatPerspectiveNormalizer(
		cv::Mat &input, std::vector<cv::Point> &outline, cv::Mat &output){
	(void) outline;
	output = input.clone();
}

/**
 * @return true, if plate is one of the known plates
 */
int megaPlateRecognisation(cv::Mat &input){
	return -1;
}
