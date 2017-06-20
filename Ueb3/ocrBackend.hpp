/*
 * ocrBackend.cpp
 *
 *  Created on: Jun 20, 2017
 *      Author: rooot
 */

#pragma once
#include "opencv2/text.hpp"
#include <iostream>
/**
 * @return true, if plate is one of the known plates
 */
std::vector<std::string> ocr(cv::Mat &grey);

//Calculate edit distance netween two words
size_t edit_distance(const std::string& A, const std::string& B);
size_t min(size_t x, size_t y, size_t z);
bool   isRepetitive(const std::string& s);
bool   sort_by_lenght(const std::string &a, const std::string &b);
//Draw ER's in an image via floodFill
void   er_draw(std::vector<cv::Mat> &channels,
		std::vector<std::vector<cv::text::ERStat> > &regions,
		std::vector<cv::Vec2i> group, cv::Mat& segmentation);
