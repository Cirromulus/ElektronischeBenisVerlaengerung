/*
 * hotShit.hpp
 *
 *  Created on: Jun 20, 2017
 *      Author: rooty
 */

#pragma once

#include "opencv2/features2d/features2d.hpp"

void tightPreprocessing(cv::Mat &img);

//Struct for storing Point and accumulated distance to origin
struct accDistanceAndPoint{
	cv::Point pt;
	int accDist;

	//overwriting operator to make sorting of accDistanceAndPoint possible by comparing accDist
	inline bool operator<(const accDistanceAndPoint& a) const{
		return  accDist < a.accDist;
	}
};

void hardSegmentation(cv::Mat &input, std::vector<cv::Point2f> &output);

cv::Mat phatPerspectiveNormalizer(cv::Mat &input, std::vector<cv::Point2f> &outline);

/**
 * @return true, if plate is one of the known plates
 */
int megaPlateRecognisificationessing(cv::Mat &input);
