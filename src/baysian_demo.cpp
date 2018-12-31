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

void ReadConfig(double& ratio, Point2i& loc, bool& save_video, bool& is_demo)
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
    is_demo = ((int)fs["baysian_bear"] == 0) ? false : true;
}


int main(int argc, char **argv) {
    double ratio = 0; 
    Point2i loc;
    bool save_video = false, is_demo = false;
    ReadConfig(ratio, loc, save_video, is_demo);
    if(is_demo)
    {
        Mat teddy = imread("../test_images/teddy.jpg");
        imshow("demo_source_image", teddy);
        Mat trimap = imread("../test_images/trimapT.png");
        imshow("trimap", trimap);
        BayesianMatting b_matting;
        Mat foreground = b_matting.Process(teddy, trimap);
        imshow("after_matting", foreground);
        waitKey(0);
    }
    else
    {
        VideoCapture capture("../videos/chicken.mp4");
        VideoCapture capture_mix("../videos/cause_way.mp4");

        TrimapGenerator tg;
        BayesianMatting b_matting;
        while (1)
        {
            int cur_time = getTickCount();
            Mat frame, frame_mix;
            capture >> frame;
            capture_mix >> frame_mix;
            // cout << frame.at<Vec3b>(0, 0) << endl;
            if (frame.empty() || frame_mix.empty()) {
                printf("video play complete\n");
                break;
            }

            resize(frame, frame, Size(), 0.25, 0.25);
            resize(frame_mix, frame_mix, Size(), 0.5, 0.5);
            Mat trimap = tg.Process(frame);
            Mat alpha = b_matting.Process(frame, trimap);
            Mat roi = frame_mix(Range(loc.y, loc.y + frame.rows), 
                        Range(loc.x, loc.x + frame.cols));
            // Mat foreground_mask;
            // cvtColor(foreground_mask, foreground_mask, COLOR_BGR2GRAY);
            // threshold(foreground_mask, foreground_mask, 1, 255, THRESH_BINARY||THRESH_OTSU);
            for(int r = 0; r < roi.rows; r ++)
            {
                for(int c = 0; c < roi.cols; c ++)
                {
                    roi.at<Vec3b>(r, c) = alpha.at<double>(r, c) * frame.at<Vec3b>(r, c) 
                                + (1 - alpha.at<double>(r, c)) * roi.at<Vec3b>(r, c);

                    // roi.at<Vec3b>(r, c)[0] = saturate_cast<uchar>(alpha.at<float>(r, c) * frame.at<Vec3b>(r, c)[0] 
                    //             + (1 - alpha.at<float>(r, c)) * roi.at<Vec3b>(r, c)[0]);
                    // roi.at<Vec3b>(r, c)[1] = saturate_cast<uchar>(alpha.at<float>(r, c) * frame.at<Vec3b>(r, c)[1] 
                    //             + (1 - alpha.at<float>(r, c)) * roi.at<Vec3b>(r, c)[1]);
                    // roi.at<Vec3b>(r, c)[2] = saturate_cast<uchar>(alpha.at<float>(r, c) * frame.at<Vec3b>(r, c)[2] 
                    //             + (1 - alpha.at<float>(r, c)) * roi.at<Vec3b>(r, c)[2]);
                }
            }

            imshow("video", frame_mix);
            waitKey(5);
        }
    }
}
