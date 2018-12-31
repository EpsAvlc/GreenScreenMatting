#ifndef BAYESIAN_MATTING_H_
#define BAYESIAN_MATTING_H_

#include <Eigen/Core>

#include <opencv2/core/core.hpp>

class BayesianMatting
{
public: 
    cv::Mat Process(const cv::Mat& ori_img, cv::Mat& tripMat);
    cv::Mat MixImage(const cv::Mat& ori_img, cv::Mat& trimap, cv::Mat& bg, cv::Point2i loc);
private:
    void calcMeanAndCovMatrix(const cv::Mat& img, const cv::Mat& mask, Eigen::Vector3d& mean_vec, Eigen::Matrix3d& cov);
    
    int iter_ = 5;
    // int ori_val = 8;
};

#endif // !BAYESIAN_MATTING_H_
