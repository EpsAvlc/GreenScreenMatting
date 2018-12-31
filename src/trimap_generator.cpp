#include "trimap_generator.h"

#include <iostream>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

Mat TrimapGenerator::Process(const cv::Mat& ori_img)
{
    vector<Mat> frame_bgrs;
    split(ori_img, frame_bgrs);
    
    Mat background_mask = Mat::zeros(ori_img.size(), CV_8UC1);
    Mat green(ori_img.size(), CV_8UC3, Scalar(43, 173, 23));
    Mat diff_img;
    absdiff(ori_img, green, diff_img);
    vector<Mat> diff_img_bgr;
    split(diff_img, diff_img_bgr);

    // low threshold to get certain backgournd mask
    Mat b_thres, g_thres, r_thres;
    threshold(diff_img_bgr[0], b_thres, 30, 255, THRESH_BINARY_INV);
    threshold(diff_img_bgr[1], g_thres, 30, 255, THRESH_BINARY_INV);
    threshold(diff_img_bgr[2], r_thres, 30, 255, THRESH_BINARY_INV);

    Mat bg;
    bitwise_and(b_thres, g_thres, bg);
    bitwise_and(bg, r_thres, background_mask);

    // high threshold to get certain foregournd mask
    Mat foreground_mask = Mat::zeros(ori_img.size(), CV_8UC1);
    threshold(diff_img_bgr[0], b_thres, 60, 255, THRESH_BINARY);
    threshold(diff_img_bgr[1], g_thres, 60, 255, THRESH_BINARY);
    threshold(diff_img_bgr[2], r_thres, 60, 255, THRESH_BINARY);

    bitwise_or(b_thres, g_thres, bg);
    bitwise_or(bg, r_thres, foreground_mask);

    Mat trimap(ori_img.size(), CV_8UC1, Scalar(128));
    Mat background_mask_inv = 255 - background_mask;
    background_mask_inv.copyTo(trimap, background_mask);
    foreground_mask.copyTo(trimap, foreground_mask);

    // imshow("background_mask", trimap);
    // waitKey(0);    

    return trimap;
}