#pragma once
#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

class TestData
{
private:
    /* data */
    cv::Mat m_color1,m_color2,m_depth1;
    cv::Mat m_K;//camera intrinsic
    std::vector<cv::KeyPoint> m_keypoints_1, m_keypoints_2;
    std::vector<cv::DMatch> m_matches;
    std::vector<cv::Point3f> m_pts_3d;
    std::vector<cv::Point2f> m_pts_2d;
    cv::Mat r_tmp,t_mat,R_mat;
    Eigen::Matrix3d R_eigen;
    Eigen::Vector3d t_eigen;
private:
    void find_feature_matches();
    void get_matched_points3D2D();
    inline cv::Point2d pix2cam(const cv::Point2d& p ){
        return cv::Point2d(
               ( p.x - m_K.at<double> ( 0,2 ) ) / m_K.at<double> ( 0,0 ),
               ( p.y - m_K.at<double> ( 1,2 ) ) / m_K.at<double> ( 1,1 )
           );
    }
    
public:
    TestData() = delete;
    TestData(const char** argv,cv::Mat _K){
        m_color1 = cv::imread( argv[1], CV_LOAD_IMAGE_COLOR);
        m_color2 = cv::imread( argv[2], CV_LOAD_IMAGE_COLOR);
        m_depth1 = cv::imread( argv[3], CV_LOAD_IMAGE_UNCHANGED );
        m_K = _K;
    }
    ~TestData(){};
    void solveRelativePose3D2D();
    std::vector<cv::Point3f> get3Dpoint(){
        return m_pts_3d;
    }
    std::vector<cv::Point2f> get2Dpoint(){
        return m_pts_2d;
    }
    Eigen::Matrix3d getRotationMatrixEigen(){return R_eigen;}
    Eigen::Vector3d getTranslationVectorEigen(){return t_eigen;}
};
void TestData::find_feature_matches(){
    using namespace cv;
    //-- 初始化
    Mat descriptors_1, descriptors_2;
    // used in OpenCV3
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    // use this if you are in OpenCV2
    // Ptr<FeatureDetector> detector = FeatureDetector::create ( "ORB" );
    // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ( "ORB" );
    Ptr<DescriptorMatcher> matcher  = DescriptorMatcher::create ( "BruteForce-Hamming" );
    //-- 第一步:检测 Oriented FAST 角点位置
    detector->detect ( m_color1,m_keypoints_1 );
    detector->detect ( m_color2,m_keypoints_2 );

    //-- 第二步:根据角点位置计算 BRIEF 描述子
    descriptor->compute ( m_color1, m_keypoints_1, descriptors_1 );
    descriptor->compute ( m_color2, m_keypoints_2, descriptors_2 );

    //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
    std::vector<DMatch> match;
    // BFMatcher matcher ( NORM_HAMMING );
    matcher->match ( descriptors_1, descriptors_2, match );

    //-- 第四步:匹配点对筛选
    double min_dist=10000, max_dist=0;

    //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        double dist = match[i].distance;
        if ( dist < min_dist ) min_dist = dist;
        if ( dist > max_dist ) max_dist = dist;
    }

    printf ( "-- Max dist : %f \n", max_dist );
    printf ( "-- Min dist : %f \n", min_dist );

    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        if ( match[i].distance <= max ( 2*min_dist, 30.0 ) )
        {
            m_matches.push_back ( match[i] );
        }
    }
    /*    Mat result;
    drawMatches(img_1,keypoints_1,img_2,keypoints_2,match,result);
    imshow("match result",result);
    waitKey(0);*/
}
void TestData::get_matched_points3D2D(){
    find_feature_matches();    
    for ( cv::DMatch m:m_matches )
    {
        ushort d = m_depth1.ptr<unsigned short> (int ( m_keypoints_1[m.queryIdx].pt.y )) [ int ( m_keypoints_1[m.queryIdx].pt.x ) ];
        if ( d == 0 )   // bad depth
            continue;
        float dd = d/5000.0;
        cv::Point2d p1 = pix2cam ( m_keypoints_1[m.queryIdx].pt );
        m_pts_3d.push_back ( cv::Point3f ( p1.x*dd, p1.y*dd, dd ) );
        m_pts_2d.push_back ( m_keypoints_2[m.trainIdx].pt );
    }
    std::cout<<"--3d-2d pairs: "<<m_pts_3d.size() <<std::endl;
}

void TestData::solveRelativePose3D2D(){
    get_matched_points3D2D();
    cv::solvePnP(m_pts_3d,m_pts_2d,m_K,cv::Mat(),r_tmp,t_mat,false,CV_EPNP);
    cv::Rodrigues(r_tmp,R_mat);
    
    cv::cv2eigen(R_mat,R_eigen);
    cv::cv2eigen(t_mat,t_eigen);
}
