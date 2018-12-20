#include <iostream>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "color_matting.h"
#include "bayesian_matting.h"

using namespace std;
using namespace cv;

Mat trimap;


int main(int argc, char **argv) {
    // string 
//     VideoCapture capture("../video/human.mp4");
//     ColorMatting matting;

//     while (1)
// 	{
//         int cur_time = getTickCount();
// 		Mat frame;
// 		capture >> frame;
//         // cout << frame.at<Vec3b>(0, 0) << endl;
// 		if (frame.empty()) {
// 			printf("complete\n");
// 			break;
// 		}

//         Mat foreground_mask = matting.Process(frame);

//         Mat foreground;
//         frame.copyTo(foreground, foreground_mask);
// 		imshow("video", foreground);
//         // cout << (getTickCount() - cur_time) << endl;
// 		waitKey(10);
// 	}
    Mat ori_img = imread("../test_images/teddy.jpg");
    trimap = imread("../test_images/trimapT.png");

    BayesianMatting b_matting;
    b_matting.Process(ori_img, trimap);


}
