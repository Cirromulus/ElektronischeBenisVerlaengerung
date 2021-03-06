/*
 * coreFunctions.hpp
 *
 *  Created on: Jun 20, 2017
 *      Author: rooty
 */

#pragma once

#include "opencv2/features2d/features2d.hpp"

void preprocessing(cv::Mat &img, bool live = false);

//Struct for storing Point and accumulated distance to origin
struct accDistanceAndPoint{
	cv::Point pt;
	int accDist;

	//overwriting operator to make sorting of accDistanceAndPoint possible by comparing accDist
	inline bool operator<(const accDistanceAndPoint& a) const{
		return  accDist < a.accDist;
	}
};

void findPlates(cv::Mat input, std::vector<cv::Point2f> &output);

cv::Mat deWarp(cv::Mat &input, std::vector<cv::Point2f> &outline);

/**
 * @return true, if plate is one of the known plates
 */
int lookupPlate(cv::Mat &input);
