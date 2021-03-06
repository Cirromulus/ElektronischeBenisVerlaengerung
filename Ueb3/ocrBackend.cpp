/*
 * ocrBackend.cpp
 *
 *  Created on: Jun 20, 2017
 *      Author: rooot
 */

#include "ocrBackend.hpp"
#include "knownPlates.hpp"
#include "helpers.hpp"
#include <stdlib.h>

#include <iostream>

using namespace std;
using namespace cv;
using namespace cv::text;

vector<string> CustomOCR::ocr(cv::Mat &grey){
    vector<Mat> channels;
    Mat color;
    cvtColor(grey, color, CV_GRAY2RGB);


    //This is our only channel, and we dont have to read inverted text
    channels.push_back(grey);
    //channels.push_back(255-grey);

    double t_d = (double)getTickCount();
    // Create ERFilter objects with the 1st and 2nd stage default classifiers


    vector<vector<ERStat> > regions(channels.size());
    // Apply the default cascade classifier to each independent channel (could be done in parallel)
    for (int c=0; c<(int)channels.size(); c++)
    {
        er_filter1->run(channels[c], regions[c]);
        er_filter2->run(channels[c], regions[c]);
    }

    if(debug) cout << "TIME_REGION_DETECTION = " << ((double)getTickCount() - t_d)*1000/getTickFrequency() << endl;

    Mat out_img_decomposition= Mat::zeros(color.rows+2, color.cols+2, CV_8UC1);
    vector<Vec2i> tmp_group;
    for (int i=0; i<(int)regions.size(); i++)
    {
        for (int j=0; j<(int)regions[i].size();j++)
        {
            tmp_group.push_back(Vec2i(i,j));
        }
        Mat tmp= Mat::zeros(color.rows+2, color.cols+2, CV_8UC1);
        er_draw(channels, regions, tmp_group, tmp);
        if (i > 0)
            tmp = tmp / 2;
        out_img_decomposition = out_img_decomposition | tmp;
        tmp_group.clear();
    }

    double t_g = (double)getTickCount();
    // Detect character groups
    vector< vector<Vec2i> > nm_region_groups;
    vector<Rect> nm_boxes;
    erGrouping(color, channels, regions, nm_region_groups, nm_boxes,ERGROUPING_ORIENTATION_HORIZ);
    if(debug) cout << "TIME_GROUPING = " << ((double)getTickCount() - t_g)*1000/getTickFrequency() << endl;


    /*Text Recognition (OCR)*/

    string output;

    Mat out_img;
    Mat out_img_detection;
    Mat out_img_segmentation = Mat::zeros(color.rows+2, color.cols+2, CV_8UC1);
    color.copyTo(out_img);
    color.copyTo(out_img_detection);
    float scale_img  = 800.f/color.cols;
    float scale_font = (float)(scale_img)/2.f;
    vector<string> words_detection;

    double t_r;
    if(debug) t_r = (double)getTickCount();

    vector<string> ret;
    for (int i=0; i<(int)nm_boxes.size(); i++)
    {

        rectangle(out_img_detection, nm_boxes[i].tl(), nm_boxes[i].br(), Scalar(0,255,255), 3);

        Mat group_img = Mat::zeros(color.rows+2, color.cols+2, CV_8UC1);
        er_draw(channels, regions, nm_region_groups[i], group_img);
        Mat group_segmentation;
        group_img.copyTo(group_segmentation);
        //image(nm_boxes[i]).copyTo(group_img);
        group_img(nm_boxes[i]).copyTo(group_img);
        copyMakeBorder(group_img,group_img,15,15,15,15,BORDER_CONSTANT,Scalar(0));

        vector<Rect>   boxes;
        vector<string> words;
        vector<float>  confidences;
        tesser->run(group_img, output, &boxes, &words, &confidences, OCR_LEVEL_WORD);

        output.erase(remove(output.begin(), output.end(), '\n'), output.end());
        if(debug) cout << "OCR Found \"" << output << "\"" << endl;
        ret.insert(ret.end(), output);

        if (output.size() < 3)
            continue;

        if(debug){
			//Some beatuiful debug images
			for (int j=0; j<(int)boxes.size(); j++)
			{
				boxes[j].x += nm_boxes[i].x-15;
				boxes[j].y += nm_boxes[i].y-15;

				//cout << "  word = " << words[j] << "\t confidence = " << confidences[j] << endl;
				if ((words[j].size() < 2) || (confidences[j] < 51) ||
						((words[j].size()==2) && (words[j][0] == words[j][1])) ||
						((words[j].size()< 4) && (confidences[j] < 60)) ||
						isRepetitive(words[j]))
					continue;
				words_detection.push_back(words[j]);
				rectangle(out_img, boxes[j].tl(), boxes[j].br(), Scalar(255,0,255),3);
				Size word_size = getTextSize(words[j], FONT_HERSHEY_SIMPLEX, (double)scale_font, (int)(3*scale_font), NULL);
				rectangle(out_img, boxes[j].tl()-Point(3,word_size.height+3), boxes[j].tl()+Point(word_size.width,0), Scalar(255,0,255),-1);
				putText(out_img, words[j], boxes[j].tl()-Point(1,1), FONT_HERSHEY_SIMPLEX, scale_font, Scalar(255,255,255),(int)(3*scale_font));
				out_img_segmentation = out_img_segmentation | group_segmentation;
			}
        }
    }

    if(debug) cout << "TIME_OCR = " << ((double)getTickCount() - t_r)*1000/getTickFrequency() << endl;

    if(debug){
		resize(out_img,out_img,Size(color.cols*scale_img,color.rows*scale_img));
		namedWindow("recognition",WINDOW_AUTOSIZE);
		imshow("recognition", out_img);
    }
    return ret;
}

size_t CustomOCR::min(size_t x, size_t y, size_t z)
{
    return x < y ? std::min(x,z) : std::min(y,z);
}

size_t CustomOCR::edit_distance(const string& A, const string& B)
{
    size_t NA = A.size();
    size_t NB = B.size();

    vector< vector<size_t> > M(NA + 1, vector<size_t>(NB + 1));

    for (size_t a = 0; a <= NA; ++a)
        M[a][0] = a;

    for (size_t b = 0; b <= NB; ++b)
        M[0][b] = b;

    for (size_t a = 1; a <= NA; ++a)
        for (size_t b = 1; b <= NB; ++b)
        {
            size_t x = M[a-1][b] + 1;
            size_t y = M[a][b-1] + 1;
            size_t z = M[a-1][b-1] + (A[a-1] == B[b-1] ? 0 : 1);
            M[a][b] = min(x,y,z);
        }

    return M[A.size()][B.size()];
}

bool CustomOCR::isRepetitive(const string& s)
{
    int count = 0;
    for (int i=0; i<(int)s.size(); i++)
    {
        if ((s[i] == 'i') ||
                (s[i] == 'l') ||
                (s[i] == 'I'))
            count++;
    }
    if (count > ((int)s.size()+1)/2)
    {
        return true;
    }
    return false;
}


void CustomOCR::er_draw(vector<Mat> &channels, vector<vector<ERStat> > &regions, vector<Vec2i> group, Mat& segmentation)
{
    for (int r=0; r<(int)group.size(); r++)
    {
        ERStat er = regions[group[r][0]][group[r][1]];
        if (er.parent != NULL) // deprecate the root region
        {
            int newMaskVal = 255;
            int flags = 4 + (newMaskVal << 8) + FLOODFILL_FIXED_RANGE + FLOODFILL_MASK_ONLY;
            floodFill(channels[group[r][0]],segmentation,Point(er.pixel%channels[group[r][0]].cols,er.pixel/channels[group[r][0]].cols),
                      Scalar(255),0,Scalar(er.level),Scalar(0),flags);
        }
    }
}

bool CustomOCR::sort_by_length(const string &a, const string &b){
	return (a.size()>b.size());
}

//-----------------------------------

LexiconOCR::LexiconOCR(){
	for(unsigned int i = 0; i < numberOfKnownPlates; i++){
		char *plate = strdup(knownPlates[i]);
		char *subelem = strtok(plate, ":");
		while(subelem) {
			lexicon.push_back(string(subelem));
			subelem = strtok(NULL, ":");
		}
	}
	if(debug){
		cout << "Lexicon: " << endl;
		for(string elem : lexicon){
			cout << "\t " << elem << endl;
		}
	}
	// must have the same order as the clasifier output classes
	vocabulary = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
	emission_p = Mat::eye(62,62,CV_64FC1);
	createOCRHMMTransitionsTable(vocabulary,lexicon,transition_p);
	ocrDecoder = OCRBeamSearchDecoder::create(
				loadOCRBeamSearchClassifierCNN("OCRBeamSearch_CNN_model_data.xml.gz"),
				vocabulary, transition_p, emission_p, OCR_DECODER_VITERBI, 50);
}

std::vector<std::string> LexiconOCR::ocr(Mat &grey){
    double t_r = (double)getTickCount();
    string output;

    vector<Rect>   boxes;
    vector<string> words;
    vector<float>  confidences;
    ocrDecoder->run(grey, output, &boxes, &words, &confidences, OCR_LEVEL_WORD);

    if(debug) cout << "OCR output = \"" << output << "\". Decoded in "
         << ((double)getTickCount() - t_r)*1000/getTickFrequency() << " ms." << endl << endl;

    if(debug){
    	for(unsigned int i = 0; i < boxes.size(); i++){
    		cout << words[i] << " conf: " << confidences[i] << endl;
    	}
    }
    return words;
}
