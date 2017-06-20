/*
 * hotShit.hpp
 *
 *  Created on: Jun 20, 2017
 *      Author: rooty
 */

#pragma once

#include "opencv2/features2d/features2d.hpp"

void tightPreprocessing(cv::Mat &img);

void hardSegmentation(cv::Mat &input, std::vector<cv::Point> &output);

void phatPerspectiveNormalizer(
		cv::Mat &input, std::vector<cv::Point> &outline, cv::Mat &output);

/**
 * @return true, if plate is one of the known plates
 */
int megaPlateRecognisation(cv::Mat &input);
