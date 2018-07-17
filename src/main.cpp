#include "bundleAdjustment.h"
#include "testData.h"

#include <opencv2/core/eigen.hpp>
using namespace cv;
using namespace std;


int main(int argc,const char** argv){
  
   if ( argc != 4 )
    {
        cout<<"usage: testG2O img1 img2 depth1"<<endl;
        return 1;
    }
    cv::Mat K = (cv::Mat_<double>(3,3)<<520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    //construct test data
    TestData testdata(argv,K);
    testdata.solveRelativePose3D2D();
    vector<cv::Point3f> pts_3d = testdata.get3Dpoint();
    vector<cv::Point2f> pts_2d = testdata.get2Dpoint();

    Eigen::Matrix3d R_eigen = testdata.getRotationMatrixEigen();
    Eigen::Vector3d t_eigen = testdata.getTranslationVectorEigen();

    Camera camera(K.at<double>(0,0),K.at<double>(1,1),K.at<double>(0,2),K.at<double>(1,2));
    
    //construct Ba problem
    BundleAdjust* BA = new BundleAdjust(pts_3d,pts_2d,camera);
    calTime tic;
    BA->construct(R_eigen,t_eigen);
    std::cout<<"Ba cost time"<<tic.toc()<<std::endl; 
    
  return 0;
}
