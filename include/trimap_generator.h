#ifndef TRIMAP_GENERATOR_H_
#define TRIMAP_GENERATOR_H_

#include <opencv2/core/core.hpp>

class TrimapGenerator
{
public:
    cv::Mat Process(const cv::Mat& ori_img);
    
};

#endif