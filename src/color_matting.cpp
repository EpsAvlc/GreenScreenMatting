#include "color_matting.h"

#include <iostream>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

Mat ColorMatting::Process(Mat& input_img)
{
    vector<Mat> frame_bgrs;
    split(input_img, frame_bgrs);
    
    Mat foreground_mask = Mat::zeros(input_img.size(), CV_8UC1);
    Mat green(input_img.size(), CV_8UC3, Scalar(43, 173, 23));
    Mat diff_img;
    absdiff(input_img, green, diff_img);
    vector<Mat> diff_img_bgr;
    split(diff_img, diff_img_bgr);

    Mat b_thres, g_thres, r_thres;
    threshold(diff_img_bgr[0], b_thres, 60, 255, THRESH_BINARY);
    threshold(diff_img_bgr[1], g_thres, 60, 255, THRESH_BINARY);
    threshold(diff_img_bgr[2], r_thres, 60, 255, THRESH_BINARY);

    Mat bg;
    bitwise_or(b_thres, g_thres, bg);
    bitwise_or(bg, r_thres, foreground_mask);
    
    Mat erodeStruct = getStructuringElement(MORPH_ELLIPSE,Size(4,4));
    erode(foreground_mask, foreground_mask, erodeStruct); 

    return foreground_mask;   
}