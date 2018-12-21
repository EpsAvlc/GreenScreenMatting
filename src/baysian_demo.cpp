#include <iostream>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "color_matting.h"
#include "bayesian_matting.h"
#include "trimap_generator.h"

using namespace std;
using namespace cv;

int main(int argc, char **argv) {
    VideoCapture capture("../videos/chicken.mp4");

    TrimapGenerator tg;
    BayesianMatting b_matting;
    while (1)
	{
        int cur_time = getTickCount();
		Mat frame;
		capture >> frame;
        // cout << frame.at<Vec3b>(0, 0) << endl;
		if (frame.empty()) {
			printf("complete\n");
			break;
		}
        resize(frame, frame, Size(), 0.25, 0.25);
        Mat trimap = tg.Process(frame);
        Mat foreground = b_matting.Process(frame, trimap);
        imshow("foreground", foreground);
        waitKey(5);
	}
}
