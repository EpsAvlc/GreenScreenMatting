#ifndef BASE_MATTING_H__
#define BASE_MATTING_H__

#include <opencv2/core/core.hpp>

class BaseMatting
{
public:
    virtual cv::Mat Process(cv::Mat& input_img) = 0;
private:
    
};

#endif // !BASE_MATTING_H__
