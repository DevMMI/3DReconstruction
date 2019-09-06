#include <iostream>
#include <Eigen/Dense>
//#include <Eigen2CV.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <fstream>
#include "kernels.h"
using namespace std;
using namespace cv;
using namespace Eigen;

template <typename T>
void out(vector<T> v){
        for(auto& i : v) {
                printf("%f ", i);
        }
}

template <typename T>
void outParent(vector<vector<T> > v){
        for(auto& i : v) {
                out(i);
                printf("\n");
        }
}

void visualize(Mat m){
        namedWindow("s", WINDOW_NORMAL);
        resizeWindow("s", Size(600,600));
        imshow("s", m);
        waitKey(0);
}

void write_as_csv(Mat m, string s){
        std::fstream outputFile;
        s += ".csv";
        outputFile.open(s, std::ios::out );
        outputFile<<format(m, Formatter::FMT_CSV) <<endl;
        outputFile.close();
}


int main()
{


  Quaterniond fl(0.4970472353380268, -0.5019101102457599, 0.4972006970559577, -0.5038072587041428 );
  Quaterniond fr(0.500194345267589, -0.4998650810783542, 0.5049370794871607, -0.4949535972592133 );
  Quaterniond diff = fr * fl.inverse();
  cout<<"quat "<<diff.w()<<diff.vec()<<endl;
  Matrix3d rot = diff.toRotationMatrix();
  cout<<"mat "<<rot<<endl;
  Mat R, dst;
  eigen2cv(rot,R);
  //dst2.convertTo(dst, CV_8UC1 );
  //write_as_csv(dst2, "quat_rot");

        cout << "Hello this is C++" << endl;
        myfunc();
        string i = "/home/mohamedisse/Documents/ComputerVision/reconstruction/samples/stereo_front_left.jpg";
        string j = "/home/mohamedisse/Documents/ComputerVision/reconstruction/samples/stereo_front_right.jpg";
        Mat image_l = imread(i, 0);
        Mat image_r = imread(j, 0);

        // calibrating image
        Mat camera_mat_fl = (Mat_<double>(3,3) << 3666.737987697073, 0.0, 1230.5215533709008, 0.0, 3666.737987697073, 1059.5814244164, 0.0, 0.0, 1.0);
        vector<float> dist_coeff_fl({ -0.10738858957839269, 0.19257998343246094, 0.0, 0.0, -0.004826963862134801 });

        Mat camera_mat_fr = (Mat_<double>(3,3) << 3663.3831779453944, 0.0, 1245.152498763069, 0.0,  3663.3831779453944, 1073.9194352453455, 0.0, 0.0, 1.0);
        vector<float> dist_coeff_fr({ -0.10579500162786715, 0.15297214630504186, 0.0, 0.0, 0.1282398068572198 });

        Mat image_fl, image_fr;

        undistort(image_l, image_fl, camera_mat_fl, dist_coeff_fl);
        undistort(image_r, image_fr, camera_mat_fr, dist_coeff_fr);

        // smoothing images
        GaussianBlur(image_fl, image_fl, Size(5,5), 0, 0);
        GaussianBlur(image_fr, image_fr, Size(5,5), 0, 0);


        //Mat R = (Mat_<double>(3,3) << -1.69452405498718E-05,  -0.0002184853,  0.0114668903, 0.0020403973, -0.0001182853,  0.0096541449, -0.0103132624,  -0.0017317883,  -2.45211266265656E-07);
        Mat t = (Mat_<double>(3,1) << 0.00161991603, -0.30062434169, -0.00096649829);

        imwrite("first_stage_l.png", image_fl);
        imwrite("first_stage_r.png", image_fr);
      //   cv::Mat R1, R2, P1, P2, map11, map12, map21, map22;
      //
      // // IF BY CALIBRATED (BOUGUET'S METHOD)
      // //
      // if (!useUncalibrated) {
      //   stereoRectify(M1, D1, M2, D2, imageSize, R, T, R1, R2, P1, P2,
      //                 cv::noArray(), 0);
      //   isVerticalStereo = fabs(P2.at<double>(1, 3)) > fabs(P2.at<double>(0, 3));
      //   // Precompute maps for cvRemap()
      //   initUndistortRectifyMap(M1, D1, R1, P1, imageSize, CV_16SC2, map11,
      //                           map12);
      //   initUndistortRectifyMap(M2, D2, R2, P2, imageSize, CV_16SC2, map21,
      //                           map22);

        Mat r1, r2, p1, p2, q, map11, map12, map21, map22;
        auto imageSize = image_l.size();
        stereoRectify(camera_mat_fl, dist_coeff_fl, camera_mat_fr, dist_coeff_fr, imageSize,  R, t, r1, r2, p1, p2, q);

        // write_as_csv(r1, "r1");
        // write_as_csv(r2, "r2");
        // write_as_csv(p1, "p1");
        // write_as_csv(p2, "p2");
        initUndistortRectifyMap(camera_mat_fl, dist_coeff_fl, r1, p1, imageSize, CV_16SC2, map11, map12);
        initUndistortRectifyMap(camera_mat_fr, dist_coeff_fr, r2, p2, imageSize, CV_16SC2, map21, map22);
        // write_as_csv(map11, "map11");
        // write_as_csv(map12, "map12");
        // write_as_csv(map21, "map21");
        // write_as_csv(map22, "map22");

        cv::Mat pair;
        pair.create(imageSize.height, imageSize.width * 2, CV_8UC3);
        Mat image_fl_r, image_fr_r;
        remap(image_l, image_fl_r, map11, map12, INTER_LINEAR);
        remap(image_r, image_fr_r, map21, map22, INTER_LINEAR);

        imwrite("rectifiedL.png", image_fl_r);
        imwrite("rectifiedR.png", image_fr_r);
        Mat disparity, disp, vdisp;


        //(int minDisparity=0, int numDisparities=16, int blockSize=3, int P1=0, int P2=0, int disp12MaxDiff=0,
        //int preFilterCap=0, int uniquenessRatio=0, int speckleWindowSize=0, int speckleRange=0, int mode=StereoSGBM::MODE_SGBM)
        Ptr<StereoSGBM> stereo = StereoSGBM::create(
                -64, 128, 11, 100, 1000, 32, 0, 15, 1000, 16, StereoSGBM::MODE_HH);

        stereo->compute(image_fl_r, image_fr_r, disp);
        cv::normalize(disp, vdisp, 0, 256, cv::NORM_MINMAX, CV_8U);
        cv::imwrite("disparity.jpg", vdisp);

        cv::Mat part = pair.colRange(0, imageSize.width);
        cvtColor(image_fl_r, part, cv::COLOR_GRAY2BGR);
        part = pair.colRange(imageSize.width, imageSize.width * 2);
        cvtColor(image_fr_r, part, cv::COLOR_GRAY2BGR);
        for (int j = 0; j < imageSize.height; j += 16)
          cv::line(pair, cv::Point(0, j), cv::Point(imageSize.width * 2, j), cv::Scalar(0, 255, 0));

        cv::imwrite("rectified.jpg", pair);



}
