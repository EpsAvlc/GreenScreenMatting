#ifndef COLOR_MATTING_H_
#define COLOR_MATTING_H_

#include "base_matting.h"

class ColorMatting : public BaseMatting
{
public:
    virtual cv::Mat Process(cv::Mat& input_img);
private:
    
};

#endif // !COLOR_MATTING_H_