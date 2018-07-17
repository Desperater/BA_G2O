#pragma once
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>


#include <g2o/core/base_binary_edge.h>		//二元边
#include <g2o/core/base_unary_edge.h>		//一元边
#include <g2o/core/base_vertex.h>
#include <g2o/types/slam3d/vertex_se3.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/sparse_block_matrix.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

#include <vector>
#include <iostream>
#include <chrono>

class Camera{
public:
  Camera(double _fx,double _fy,double _cx,double _cy){
    fx = _fx;
    fy = _fy;
    cx = _cx;
    cy = _cy;
  }
  
  double cx,cy,fx,fy;
  Eigen::Vector2d cam2pixel(const Eigen::Vector3d& pos,const g2o::SE3Quat& T){
    Eigen::Matrix3d m_R = T.rotation().toRotationMatrix();
    Eigen::Vector3d m_T(T.translation());
    Eigen::Vector3d pos_sub = m_R * pos + m_T;
    
    double u = pos_sub(0)/pos_sub(2)*fx+cx;
    double v = pos_sub(1)/pos_sub(2)*fy+cy;
    return Eigen::Vector2d(u,v);
  }
  cv::Point2d pix2cam(const cv::Point2d& input){
    return cv::Point2d(
      (input.x - cx)/fx,
      (input.y - cy)/fy
    );
  }
};
//每个位姿的李代数表达构成一个节点
class BaVertexSE3:public g2o::BaseVertex<6,g2o::SE3Quat>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
BaVertexSE3(){}
  virtual void setToOriginImpl(){
    _estimate = g2o::SE3Quat();
  }
  virtual void oplusImpl(const double* update){
    Eigen::Map<const g2o::Vector6> _update(update);
    _estimate  = g2o::SE3Quat::exp(_update)*_estimate;    
  }
  virtual bool read(std::istream& in){}
  virtual bool write(std::ostream& out)const{}
  
};
class BaVertexPt3:public g2o::BaseVertex<3,Eigen::Vector3d>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
BaVertexPt3(){}
  virtual void setToOriginImpl(){
    _estimate = Eigen::Vector3d::Zero();
  }
  virtual void oplusImpl(const double * update){
    Eigen::Vector3d _update(_update);
    _estimate += _update;    
  }
  virtual bool read(std::istream& in){}
  virtual bool write(std::ostream& out)const{}
};

class BaEdge:public g2o::BaseBinaryEdge<2,Eigen::Vector2d,BaVertexPt3,BaVertexSE3>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  BaEdge(const Camera& _camera):m_camera(_camera)
  {}
  virtual void computeError(){
    const BaVertexPt3* v1 = static_cast<BaVertexPt3*>(_vertices[0]);
    const BaVertexSE3* v2 = static_cast<BaVertexSE3*>(_vertices[1]);
    Eigen::Vector2d  obs(_measurement);
    //这里是观测值减去预测值，观测值与预测值的顺序会影响雅克比矩阵的正负
    //std::cout<<"Obs is \n"<<obs<<std::endl<<"predict is \n"<<m_camera.cam2pixel(v1->estimate(),v2->estimate())<<std::endl;
    _error = obs - m_camera.cam2pixel(v1->estimate(),v2->estimate());
  }
  virtual void linearizeOplus(){

    BaVertexPt3* v1 = static_cast<BaVertexPt3*>(_vertices[0]);
    BaVertexSE3* v2 = static_cast<BaVertexSE3*>(_vertices[1]);
    g2o::SE3Quat T(v2->estimate());
    Eigen::Vector3d xyz(v1->estimate()[0],v1->estimate()[1],v1->estimate()[2]);
    Eigen::Vector3d xyz_trans(T.map(v1->estimate())[0],T.map(v1->estimate())[1],T.map(v1->estimate())[2]);
    //std::cout<<"XYZ"<<xyz<<std::endl;
   // std::cout<<"XYZ_TRANS"<<xyz_trans<<std::endl;
    
    double x = v1->estimate()[0];
    double y = v1->estimate()[1];
    double z = v1->estimate()[2];
    double z_inv = 1.0/z;
    double z_inv2 = z_inv*z_inv;
    
    Eigen::Matrix<double,2,3> jaco_uv2pos;	//像素点误差对空间点位置的偏导矩阵
    
    jaco_uv2pos(0,0) = m_camera.fx*z_inv;
    jaco_uv2pos(0,1) = 0.;
    jaco_uv2pos(0,2) = -m_camera.fx*z_inv2*x;
    
    jaco_uv2pos(1,0) = 0.;
    jaco_uv2pos(1,1) = m_camera.fy * z_inv;
    jaco_uv2pos(1,2) = -m_camera.fy *z_inv2*y;
    
    _jacobianOplusXi = -1*jaco_uv2pos * T.rotation().toRotationMatrix();
    
    _jacobianOplusXj(0,0) =  m_camera.fx*x*y*z_inv2;
    _jacobianOplusXj(0,1) = -m_camera.fx + m_camera.fx*x*x*z_inv2;
    _jacobianOplusXj(0,2) = m_camera.fx * y *z_inv;
    _jacobianOplusXj(0,3) = -m_camera.fx*z_inv;
    _jacobianOplusXj(0,4) = 0.0;
    _jacobianOplusXj(0,5) = m_camera.fx * z_inv2* x;
    
    _jacobianOplusXj(1,0) = m_camera.fy-m_camera.fy*z_inv2*y*y;
    _jacobianOplusXj(1,1) = -m_camera.fy *x*y*z_inv2;
    _jacobianOplusXj(1,2) = -m_camera.fy*z_inv*x;
    _jacobianOplusXj(1,3) = 0.0;
    _jacobianOplusXj(1,4) = -m_camera.fy*z_inv;
    _jacobianOplusXj(1,5) = m_camera.fy*z_inv2*y;    
  }
  bool read(std::istream& in){}
  bool write(std::ostream& out)const{}  
  
public:
  Camera m_camera;
};

class BundleAdjust{
public:
  BundleAdjust()=delete;
  BundleAdjust(const std::vector<cv::Point3f>& _pts3d,const std::vector<cv::Point2f>& _pts2d,const Camera& _camera)
  :m_pts3D(_pts3d),m_pts2D(_pts2d)  ,m_camera(_camera)
  {
        //sparse solver
     m_solverType = new g2o::LinearSolverCSparse<Block::PoseMatrixType>();
     //dense solver
     //m_solverType = new g2o::LinearSolverDense<Block::PoseMatrixType>();
     m_solver_ptr = new Block(std::unique_ptr<Block::LinearSolverType>(m_solverType));
     m_algorithm_type = new g2o::OptimizationAlgorithmLevenberg(std::unique_ptr<Block>(m_solver_ptr));
   
    
     m_optimizer.setAlgorithm(m_algorithm_type);
  };
  
  std::vector<cv::Point3f> getCorrectedPt3()
  {
    std::vector<cv::Point3f> corrtedPoint;
    int index = 1;
    for(int i= 0;i<m_pts3D.size();++i){
      BaVertexPt3* pt_v = dynamic_cast<BaVertexPt3*>(m_optimizer.vertex(index));
      Eigen::Vector3d pt = pt_v->estimate();
      corrtedPoint.push_back(cv::Point3f(pt(0),pt(1),pt(2)));
    }
    return corrtedPoint;
  }
  Eigen::Matrix3d getRefinedRotation(){
    return refined_R;
  }
  Eigen::Vector3d getRefinedTranslation(){
    return refined_t;
  }
  bool construct(Eigen::Matrix3d _R = Eigen::Matrix3d::Identity(),Eigen::Vector3d _t=Eigen::Vector3d::Zero()){
    

    
    BaVertexSE3* pose = new BaVertexSE3();
    pose->setId(0);
    pose->setEstimate(g2o::SE3Quat(_R,_t));
    m_optimizer.addVertex(pose);
    
    int index = 1;
    for(auto& pt:m_pts3D){
      BaVertexPt3* pos = new BaVertexPt3();
      pos->setId(index++);
      pos->setEstimate(Eigen::Vector3d(pt.x,pt.y,pt.z));
      pos->setMarginalized(true);
      m_optimizer.addVertex(pos);
    }
    
    index =1;
    for(auto& pt:m_pts2D){
      BaEdge* edge = new BaEdge(m_camera);
      edge->setId(index);
      edge->setVertex(0,dynamic_cast<BaVertexPt3*>(m_optimizer.vertex(index)));
      edge->setVertex(1,pose);
      edge->setMeasurement(Eigen::Vector2d(pt.x,pt.y));
      edge->setParameterId(0,0);
      edge->setInformation(Eigen::Matrix2d::Identity());
      
      m_optimizer.addEdge(edge);
      index++;
    }
    
    m_optimizer.setVerbose(true);
    m_optimizer.initializeOptimization();
    m_optimizer.optimize(100);	//参数表示最大迭代次数
    
    //optimization result
    
    std::cout<<"before optimization R is \n" <<_R<<std::endl<<"t is \n"<<_t<<std::endl;
    Eigen::Isometry3d T(pose->estimate());
    refined_R  = T.rotation();
    refined_t  = T.translation();
    std::cout<<"After optimization R is\n"<<T.rotation()<<std::endl<<"t is \n"<<T.translation()<<std::endl;
  }
  
private:
  typedef g2o::BlockSolver<g2o::BlockSolverTraits<6,3>> Block;
  Block::LinearSolverType* m_solverType;
  Block* m_solver_ptr;
  g2o::OptimizationAlgorithmLevenberg* m_algorithm_type;
  g2o::SparseOptimizer m_optimizer;
  Camera m_camera;
  std::vector<cv::Point3f> m_pts3D;
  std::vector<cv::Point2f> m_pts2D;
  Eigen::Matrix3d refined_R;
  Eigen::Vector3d refined_t;
};

class calTime{
public:
  calTime(){
    tic();
  }
  void tic(){
    start = std::chrono::system_clock::now();
  }
  double toc(){
    end = std::chrono::system_clock::now();
    double duration = std::chrono::duration<double>(end - start).count();
    return duration*1000;
  }
private:
  std::chrono::time_point<std::chrono::system_clock> start;
  std::chrono::time_point<std::chrono::system_clock> end;
};