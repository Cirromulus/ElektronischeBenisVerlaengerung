/*
 * helpers.hpp
 *
 *  Created on: Jun 20, 2017
 *      Author: rooty
 */

#pragma once

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <string>
#include <vector>


extern bool debug;

// The images are very large, hence for displaying they are scaled down to
// the size below.
static const int windowWidth = 1280, windowHeight = 720;

// Scales display down to windowWidth*windowHeight and shows it in
// windowName.
void showScaled (std::string windowName, cv::Mat& display);

static const cv::Scalar color[] =
  {
    CV_RGB (255, 0, 0), CV_RGB (0, 255, 0), CV_RGB (255, 255, 0), CV_RGB (0, 0, 255), CV_RGB (255, 0, 255), CV_RGB (0, 255, 255)
  };

void drawHull(cv::Mat& dst, std::vector<cv::Point2i> hull, int col);

/**
 * @brief draws a std::vector of cv::Points in given size in cycling colors
 */
void drawApprox(cv::Mat& dst, std::vector<cv::Point> hull, int col, int size);
/**
 * @brief draws a std::vector of std::vector of cv::Points in given size in cycling colors
 */
void drawApproxes(cv::Mat& dst, std::vector<std::vector<cv::Point> > app, int size = 1);
void drawRect(cv::Mat& dst, cv::RotatedRect& rec, int i);
void drawRects(cv::Mat& dst, std::vector<cv::RotatedRect>& rects);

// Draws a cross at p of given size (width=height) and color
void drawCross (cv::Mat& display, cv::Point p, int size, cv::Scalar color);

std::string type2str(int type);

// Makes a RGB image darker by 1/4
void makeImageDarker (cv::Mat& display);
/**
 * @brief Adjusts contrast and brightness to the MAX
 */
void preprocessColors(cv::Mat &img, double highestValue);
