#include <iostream>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <Eigen2CV.h>
#include <fstream>

using namespace std;
using namespace cv;
using namespace Eigen;
using namespace octane;

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
    cv::Ptr<Feature2D> sift = xfeatures2d::SIFT::create();

    //-- Step 1: Detect the keypoints:
    std::vector<KeyPoint> keypoints_1, keypoints_2;
    sift->detect( smoothed_fl, keypoints_1 );
    sift->detect( smoothed_fr, keypoints_2 );

    // sample best keypoints
    //KeyPointsFilter::retainBest(keypoints_1, 50);
    //KeyPointsFilter::retainBest(keypoints_2, 50);

    //cout<<"rows "<<keypoints_1.size()<<", cols "<<keypoints_2.size()<<endl;

    //-- Step 2: Calculate descriptors (feature vectors)
    Mat descriptors_1, descriptors_2;
    sift->compute( smoothed_fl, keypoints_1, descriptors_1 );
    sift->compute( smoothed_fr, keypoints_2, descriptors_2 );

    //cout<<"rows "<<descriptors_1.rows<<", cols "<<descriptors_1.cols<<endl;
    //-- Step 3: Matching descriptor vectors using BFMatcher :
    BFMatcher matcher;
    std::vector< DMatch > matches;
    matcher.match( descriptors_1, descriptors_2, matches );

    vector<Point2f> first_img_pts;
    vector<Point2f> second_img_pts;
    for(auto& match : matches){
      first_img_pts.push_back(keypoints_1[match.queryIdx].pt); // keypoints_1
      second_img_pts.push_back(keypoints_2[match.trainIdx].pt); // keypoints_2
    }
    Mat ransac_mask;
    Mat fund = findFundamentalMat(first_img_pts, second_img_pts, FM_RANSAC, 3.0, 0.99, ransac_mask);

    cout<<"first image pts "<<first_img_pts.size()<<endl;
    cout<<"ransac mask rows "<<ransac_mask.rows<<", cols "<<ransac_mask.cols;
    write_as_csv(ransac_mask);
    // draw matches
    //Mat matched_img;
    //drawMatches(smoothed_fl, keypoints_1, smoothed_fr, keypoints_2, matches, matched_img);

    // visualize
    //write_as_csv(edges);
    //visualize(edges);
    //imwrite("matched.jpg", matched_img);
    //imwrite("eg2.jpg", undistorted_fr);



}
