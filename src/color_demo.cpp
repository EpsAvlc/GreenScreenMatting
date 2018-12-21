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

int main(int argc, char **argv) 
{
    VideoCapture capture("../videos/human.mp4");
    VideoCapture capture_mix("../videos/cause_way.mp4");
    ColorMatting matting;

    while (1)
	{
		Mat frame, frame_mix;
		capture >> frame;
        capture_mix >> frame_mix;
		if (frame.empty() || frame_mix.empty()) {
			printf("video play complete\n");
			break;
		}

        resize(frame, frame, Size(), 0.25, 0.25);
        resize(frame_mix, frame_mix, Size(), 0.5, 0.5);
        Mat foreground_mask = matting.Process(frame);

        // Mat foreground;
        
        Mat roi = frame_mix(Range(frame_mix.rows - frame.rows, frame_mix.rows), 
                    Range(frame_mix.cols - frame.cols, frame_mix.cols));
        
        frame.copyTo(roi, foreground_mask);
		imshow("video", frame_mix);


		waitKey(10);
	}
}