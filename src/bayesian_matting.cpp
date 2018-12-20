#include "bayesian_matting.h"

#include <iostream>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <Eigen/LU>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>

using namespace cv;
using namespace std;

Mat BayesianMatting::Process(const Mat& ori_img, Mat& trimap)
{
    long cur_time = getTickCount();
    if(trimap.channels() == 3)
        cvtColor(trimap, trimap, CV_BGR2GRAY);

    Mat foreground_mask;
    threshold(trimap, foreground_mask, 200, 255, THRESH_BINARY);
    Mat foreground;
    ori_img.copyTo(foreground, foreground_mask);

    Mat background_mask;
    threshold(trimap, background_mask, 20, 255, THRESH_BINARY_INV);
    Mat background;
    ori_img.copyTo(background, background_mask);

    Mat uncertain_mask(trimap.size(), CV_8UC1, Scalar(255));
    bitwise_or(foreground_mask, background_mask, uncertain_mask);
    bitwise_not(uncertain_mask, uncertain_mask);
    Mat uncertain;
    ori_img.copyTo(uncertain, uncertain_mask);

    Eigen::Matrix3d foreground_cov;
    Eigen::Vector3d foreground_mean;
    calcMeanAndCovMatrix(foreground, foreground_mask, foreground_mean, foreground_cov);
    imshow("foreground_before", foreground);
    Eigen::Matrix3d background_cov;
    Eigen::Vector3d background_mean;
    calcMeanAndCovMatrix(background, background_mask, background_mean, background_cov);

    Eigen::Matrix3d background_cov_inv = background_cov.inverse();
    Eigen::Matrix3d foreground_cov_inv = foreground_cov.inverse();

    Mat trimap_double;
    trimap.convertTo(trimap_double, DataType<double>::type);
    Mat uncertain_alpha_mat;
    
    Mat standard_deviation;
    meanStdDev(ori_img, noArray(), standard_deviation);
    double sigma_c = 8;

    uncertain_alpha_mat = trimap_double / 255;
    int disp_num = 0;
    for(int r = 0; r < uncertain_mask.rows; r++)
    {
        uchar* cur_row = uncertain_mask.ptr<uchar>(r);
        // Vec3b* cur_row_foreground = foreground.ptr<Vec3b>(r);
        // double* cur_row_uncertain_mast = uncertain_mask.ptr<double>(r);
        for(int c = 0; c < uncertain_mask.cols; c++)
        {
            // Only cares those who's alpha is uncertain
            if(cur_row[c] == 0)
                continue;

            // cout << "I am here." << endl;
            double alpha = 0; 
            int neighbor_count = 0;
            for(int i = -1; i <= 1; i++)
                for(int j = -1; j <= 1; j++)
                {
                    if(r + i < 0 || c + i < 0 
                        || r+i >= uncertain_mask.rows || c + i >= uncertain_mask.cols)
                    {
                        continue;
                    }
                    if(r == 0 && c == 0)
                    {
                        continue;
                    }
                    neighbor_count ++;
                    alpha += uncertain_alpha_mat.at<double>(r + i, c + j);
                }
            alpha = alpha / neighbor_count;
            int pre_alpha = alpha;
            Eigen::Vector3d tmp_foreground;
            Eigen::Vector3d tmp_background;
            for(int i = 0; i < iter_; i++)
            {
                Eigen::Matrix3d UL = foreground_cov_inv + Eigen::Matrix3d::Identity() * alpha * alpha / (sigma_c * sigma_c);
                Eigen::Matrix3d UR = Eigen::Matrix3d::Identity() * alpha * (1 - alpha) / (sigma_c * sigma_c);
                Eigen::Matrix3d DL = UR;
                Eigen::Matrix3d DR = background_cov_inv + Eigen::Matrix3d::Identity() * (1 - alpha) * (1 - alpha) / (sigma_c * sigma_c);

                Eigen::MatrixXd A(6, 6);
                A.block<3, 3>(0, 0) = UL;
                A.block<3, 3>(0, 3) = UR;
                A.block<3, 3>(3, 0) = DL;
                A.block<3, 3>(3, 3) = DR;

                Eigen::Vector3d C(uncertain.at<Vec3b>(r, c)[0], uncertain.at<Vec3b>(r, c)[1], uncertain.at<Vec3b>(r, c)[2]); 
                Eigen::Vector3d BU = foreground_cov_inv * foreground_mean + C * alpha / (sigma_c * sigma_c);
                Eigen::Vector3d BD = background_cov_inv * background_mean + C * (1 - alpha) / (sigma_c * sigma_c);
                Eigen::MatrixXd B(6, 1);
                B.block<3, 1>(0, 0) = BU;
                B.block<3, 1>(3, 0) = BD;

                Eigen::MatrixXd x(6, 1);
                x = A.lu().solve(B); 
                tmp_foreground = x.block<3, 1>(0, 0);
                tmp_background = x.block<3, 1>(3, 0);
                alpha = (C - tmp_background).dot(tmp_foreground - tmp_background) 
                    / ((tmp_foreground - tmp_background).norm() * (tmp_foreground - tmp_background).norm());
                if(abs(alpha - pre_alpha) < 0.001)
                    break;
                pre_alpha = alpha;
            }
            auto cur_pixel = alpha * tmp_foreground;
            foreground.at<Vec3b>(r, c) = Vec3b(saturate_cast<uchar>(cur_pixel.x()),
                                    saturate_cast<uchar>(cur_pixel.y()),
                                    saturate_cast<uchar>(cur_pixel.z()));
            uncertain_alpha_mat.at<double>(r, c) = alpha;         
        }
    }

    cout << "Elasped " << (getTickCount() - cur_time) / getTickFrequency() << " s." << endl;
    imshow("foreground_after", foreground);
    imshow("trimap", uncertain_alpha_mat);
    waitKey(0);
}

void BayesianMatting::calcMeanAndCovMatrix(const cv::Mat& img, const cv::Mat& mask, Eigen::Vector3d& mean_vec, Eigen::Matrix3d& cov)
{
    Mat mean_mat;
    Mat std_mat;
    meanStdDev(img, mean_mat, std_mat, mask);
    mean_vec =  Eigen::Vector3d(mean_mat.at<double>(0, 0), mean_mat.at<double>(1, 0), mean_mat.at<double>(2, 0));
    cov = Eigen::Matrix3d::Zero();
    int non_zero_pixel_count = 0;;
    for(int r = 0; r < img.rows; r++)
    {
        const Vec3b* data = img.ptr<Vec3b>(r);
        for(int c = 0; c < img.cols; c++)
        {
            if(data[c][0] == 0 && data[c][1] == 0 && data[c][2] == 0)
            {
                continue;
            }
            non_zero_pixel_count++;
            // Mat cur_mat = (Mat_<uchar>(3, 1) << data[c][0], data[c][1], data[c][2]);
            Eigen::Vector3d cur_pixel(data[c][0], data[c][1], data[c][2]);
            Eigen::Vector3d offset = cur_pixel - mean_vec;
            cov = cov + offset * offset.transpose();
        }
    }
    cov = cov / non_zero_pixel_count;
}