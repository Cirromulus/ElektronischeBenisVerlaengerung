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
#include <stdio.h>

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
    cvtColor(input,output,COLOR_RGB2GRAY);
}

/**
 * @return true, if plate is one of the known plates
 */
int megaPlateRecognisation(cv::Mat &input){
	if(type2str(input.type()) != string("8UC1")){
		cout << "Converting input image to grey." << endl;
		cvtColor(input, input, COLOR_RGB2GRAY);
	}
	std::vector<std::string> texts = ocr(input);
	int found = -1;
	for(auto text : texts){
		cout << text << endl;
		for(unsigned int i = 0; i < numberOfKnownPlates; i++){
			char *plate = strdup(knownPlates[i]);
			char *subelem = strtok(plate, ":");
			bool valid = true;
			while(subelem) {
				if(!strstr(text.c_str(), subelem)){
					//Not found
					valid = false;
					break;
				}
				subelem = strtok(NULL, ":");
			}
			if(valid){
				found = i;
			}
		}
		if(found >= 0){
			break;
		}
	}
	return found;
}
