#include <iostream>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <fstream>
#include "kernels.h"
using namespace std;
using namespace cv;
using namespace Eigen;

template <typename T>
void out(vector<T> v){
  for(auto& i : v){
    printf("%f ", i);
  }
}

template <typename T>
void outParent(vector<vector<T>> v){
  for(auto& i : v){
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

void write_as_csv(Mat m){
  std::fstream outputFile;
  outputFile.open("matrix.csv", std::ios::out ) ;
  outputFile<<format(m, Formatter::FMT_CSV) <<endl;
  outputFile.close();
}


int main()
{
    cout << "Hello this is C++" << endl;
    myfunc();
    string i = "/home/mohamedisse/Documents/ComputerVision/reconstruction/samples/stereo_front_left.jpg";
    string j = "/home/mohamedisse/Documents/ComputerVision/reconstruction/samples/stereo_front_right.jpg";
    Mat input_fl = imread(i, 1);
    Mat input_fr = imread(j, 1);

    // calibrating image
    Mat camera_mat_fl = (Mat_<double>(3,3) << 3666.737987697073, 0.0, 1230.5215533709008, 0.0, 3666.737987697073, 1059.5814244164, 0.0, 0.0, 1.0);
    vector<float> dist_coeff_fl({ -0.10738858957839269, 0.19257998343246094, 0.0, 0.0, -0.004826963862134801 });

    Mat camera_mat_fr = (Mat_<double>(3,3) << 3663.3831779453944, 0.0, 1245.152498763069, 0.0, 	3663.3831779453944,	1073.9194352453455, 0.0, 0.0, 1.0);
    vector<float> dist_coeff_fr({ -0.10579500162786715, 0.15297214630504186, 0.0, 0.0, 0.1282398068572198 });

    Mat undistorted_fl;
    Mat undistorted_fr;
    undistort(input_fl, undistorted_fl, camera_mat_fl, dist_coeff_fl);
    undistort(input_fr, undistorted_fr, camera_mat_fr, dist_coeff_fr);

    Mat undistorted_fl_g;
    Mat undistorted_fr_g;
    cvtColor(undistorted_fl, undistorted_fl_g, COLOR_BGR2GRAY);
    cvtColor(undistorted_fr, undistorted_fr_g, COLOR_BGR2GRAY);

    // smoothing images
    Mat smoothed_fl;
    Mat smoothed_fr;
    GaussianBlur(undistorted_fl_g, smoothed_fl, Size(5,5), 0, 0);
    GaussianBlur(undistorted_fr_g, smoothed_fr, Size(5,5), 0, 0);

    // finding good correspondences
    // Mat edges;
    // Canny(undistorted_fr, edges, 255/3, 255, 3, true);
    std::vector<KeyPoint> keypoints_1, keypoints_2;
    Mat descriptors_1, descriptors_2;
    if(true){
      // detect SIFT keypoints
      Ptr<Feature2D> sift = xfeatures2d::SIFT::create();

      // detect using sift features
      sift->detect( smoothed_fl, keypoints_1 );
      sift->detect( smoothed_fr, keypoints_2 );

      //-- Calculate descriptors (feature vectors)
      sift->compute( smoothed_fl, keypoints_1, descriptors_1 );
      sift->compute( smoothed_fr, keypoints_2, descriptors_2 );
    }

    else{
      // detect SURF keypoints
      printf("using SURF\n");
      int minHessian = 400;

      Ptr<xfeatures2d::SURF> detector = xfeatures2d::SURF::create( minHessian );

      detector->detectAndCompute( smoothed_fl, noArray(), keypoints_1, descriptors_1 );
      detector->detectAndCompute( smoothed_fr, noArray(), keypoints_2, descriptors_2 );
    }

    // sample best keypoints
    KeyPointsFilter::removeDuplicated(keypoints_1);
    KeyPointsFilter::removeDuplicated(keypoints_2);

    //cout<<"rows "<<keypoints_1.size()<<", cols "<<keypoints_2.size()<<endl;



    //cout<<"rows "<<descriptors_1.rows<<", cols "<<descriptors_1.cols<<endl;
    //-- Step 3: Matching descriptor vectors using BFMatcher :
    vector<DMatch> matches;
    BFMatcher matcher;
    if(true){
      // knn matcher
      vector<vector<DMatch>> vect_matches;
      matcher.knnMatch( descriptors_1, descriptors_2, vect_matches, 2 );

      for (int i = 0; i < vect_matches.size(); ++i){
            const float ratio = 0.8; // As in Lowe's paper; can be tuned
            if (vect_matches[i][0].distance < ratio * vect_matches[i][1].distance)
            {
                matches.push_back(vect_matches[i][0]);
            }
      }

    }
    else{
      matcher.match( descriptors_1, descriptors_2, matches );
    }
    vector<Point2f> first_img_pts;
    vector<Point2f> second_img_pts;
    for(auto& match : matches){
      first_img_pts.push_back(keypoints_1[match.queryIdx].pt); // keypoints_1
      second_img_pts.push_back(keypoints_2[match.trainIdx].pt); // keypoints_2
    }

    Mat ransac_mask;
    Mat fund = findFundamentalMat(first_img_pts, second_img_pts, FM_RANSAC, 0.1, 0.9999, ransac_mask);

    cout<<"first image pts "<<first_img_pts.size()<<endl;
    cout<<"ransac mask rows "<<ransac_mask.rows<<", cols "<<ransac_mask.cols;
    write_as_csv(fund);
    // draw matches
    //Mat matched_img;

    std::vector< DMatch > matches_filtered;
    std::vector<KeyPoint> keypoints_1_filt, keypoints_2_filt;
    int it = 0;
    for(int i = 0; i < first_img_pts.size(); i++){
      if(ransac_mask.at<double>(i, 1) ){
        KeyPoint k1(first_img_pts[i], 1.0);
        KeyPoint k2(second_img_pts[i], 1.0);
        keypoints_1_filt.push_back(k1);
        keypoints_2_filt.push_back(k2);

        DMatch d(it, it, 1);
        matches_filtered.push_back(d);
        it++;
      }
    }
    //Mat ransac_matched_img;

    //drawMatches(smoothed_fl, keypoints_1_filt, smoothed_fr, keypoints_2_filt, matches_filtered, ransac_matched_img);
    Mat R = (Mat_<double>(3,3) << -1.69452405498718E-05,	-0.0002184853,	0.0114668903, 0.0020403973,	-0.0001182853,	0.0096541449, -0.0103132624,	-0.0017317883,	-2.45211266265656E-07);
    Mat t = (Mat_<double>(3,1) << 0.00161991603, -0.30062434169, -0.00096649829);
    //vector<double> T = {0.00161991603, -0.30062434169, -0.00096649829};

    Mat r1, r2, p1, p2, q;
    stereoRectify(camera_mat_fl, dist_coeff_fl, camera_mat_fr, dist_coeff_fr, cv::Size(2056, 2464),  R, t, r1, r2, p1, p2, q);

    Mat l_rectified, r_rectified;
    undistortPoints(input_fl, l_rectified, camera_mat_fl, dist_coeff_fl, r1, p1);
    undistortPoints(input_fr, r_rectified, camera_mat_fr, dist_coeff_fr, r2, p2);

    //Mat pt = Mat(1,3, CV_64F, float(0));
    //pt.at<float>(0,0) =
    //void cv::multiply( cv::InputArray src1, // First input array cv::InputArray src2, // Second input array cv::OutputArray dst, // Result array double scale = 1.0, // overall scale factor int dtype = -1 // Output type for result array);

    // visualize
    //write_as_csv(edges);
    //visualize(edges);
    //imwrite("ransac_matched_img_knn_sift_narrow.jpg", ransac_matched_img);
    imwrite("l_rectified.jpg", l_rectified);
    imwrite("r_rectified.jpg", r_rectified);



}
