#include <iostream>
#include <string>

#include "opencv2/opencv.hpp"

#include "color_matting.h"
#include "bayesian_matting.h"
#include "trimap_generator.h"

using namespace std;
using namespace cv;

void ReadConfig(double ratio, Point2i& loc, bool save_video)
{
    FileStorage fs("../config/config.yml", FileStorage::READ);
    assert(fs.isOpened());
    fs["ratio"] >> ratio;
    // cout << ratio << endl;
    int tmp_x, tmp_y;
    fs["img_loc_x"] >> tmp_x;
    fs["img_loc_y"] >> tmp_y;
    if(tmp_x < 0 || tmp_y < 0)
    {
        loc.x = 0;
        loc.y = 0;
    }
    else
    {
        loc.x = tmp_x;
        loc.y = tmp_y;
    }
    save_video = ((int)fs["save_video"] == 0) ? false : true;
    cout << save_video << endl;
    cout << loc << endl;
}

int main(int argc, char **argv) 
{
    VideoCapture capture("../videos/human.mp4");
    VideoCapture capture_mix("../videos/cause_way.mp4");
    ColorMatting matting;

    double ratio;
    Point2i loc;
    bool save_video;
    ReadConfig(ratio, loc, save_video);
    while(1);

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