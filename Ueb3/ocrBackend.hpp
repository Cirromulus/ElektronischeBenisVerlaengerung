/*
 * ocrBackend.cpp
 *
 *  Created on: Jun 20, 2017
 *      Author: rooot
 */

#pragma once
#include "opencv2/text.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

using namespace std;
using namespace cv;
using namespace cv::text;

class CustomOCR{
	Ptr<text::ERFilter> er_filter1;
	Ptr<text::ERFilter> er_filter2;
	Ptr<text::OCRTesseract> tesser;
public:
	CustomOCR(){
	    er_filter1 = createERFilterNM1(
	    		text::loadClassifierNM1("trained_classifierNM1.xml"),8,0.00015f,0.13f,0.2f,true,0.1f);
	    er_filter2 = createERFilterNM2(
	    		text::loadClassifierNM2("trained_classifierNM2.xml"),0.5);
	    tesser = text::OCRTesseract::create();
	}

std::vector<std::string> ocr(Mat &grey);

private:
//Calculate edit distance netween two words
size_t edit_distance(const std::string& A, const std::string& B);
size_t min(size_t x, size_t y, size_t z);
bool   isRepetitive(const std::string& s);
bool   sort_by_length(const std::string &a, const std::string &b);
//Draw ER's in an image via floodFill
void   er_draw(std::vector<Mat> &channels,
		std::vector<std::vector<text::ERStat> > &regions,
		std::vector<Vec2i> group, Mat& segmentation);
};

class LexiconOCR{
	Ptr<OCRBeamSearchDecoder> ocrDecoder;
	Mat transition_p;
	string vocabulary;
	vector<string> lexicon;  // a list of words expected to be found on the input image
	Mat emission_p;
public:
    LexiconOCR();
    std::vector<std::string> ocr(Mat &grey);
};
