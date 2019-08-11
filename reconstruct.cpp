#include <iostream>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <Eigen2CV.h>
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

int main()
{
    cout << "Hello this is C++" << endl;

    string i = "/home/mohamedisse/Documents/ComputerVision/argoverse_dataset/argoverse-tracking/sample/c6911883-1843-3727-8eaa-41dc8cda8993/stereo_front_left/stereo_front_left_315978406299256520.jpg";
    string j = "/home/mohamedisse/Documents/ComputerVision/argoverse_dataset/argoverse-tracking/sample/c6911883-1843-3727-8eaa-41dc8cda8993/stereo_front_left/stereo_front_left_315978406499056520.jpg";
    Mat input_a = imread(i, 1);
    Mat input_b = imread(j, 1);

    // calibrating image
    Mat camera_mat = (Mat_<double>(3,3) << 3666.737987697073, 0.0, 1230.5215533709008, 0.0, 3666.737987697073, 1059.5814244164, 0.0, 0.0, 1.0);
    vector<float> dist_coeff({ -0.10738858957839269, 0.19257998343246094, 0.0, 0.0, -0.004826963862134801 });

    Mat undistorted_a;
    Mat undistorted_b;
    undistort(input_a, undistorted_a, camera_mat, dist_coeff);
    undistort(input_b, undistorted_b, camera_mat, dist_coeff);


    // finding good correspondences
    Mat edges;
    canny()

    // imwrite("eg1.jpg", output_a);
    // imwrite("eg2.jpg", output_b);
    //
    // imwrite("og1.jpg", input_a);
    // imwrite("og2.jpg", input_b);


}
