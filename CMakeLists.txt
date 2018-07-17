cmake_minimum_required(VERSION 2.8)
project("BA_G2O")

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_BUILD_TYPE "Release")

find_package(G2O REQUIRED)
find_package(OpenCV 3.1 REQUIRED)
find_package(Cholmod REQUIRED)
find_package(CSparse REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(
	${PROJECT_SOURCE_DIR}/include
	${OpenCV_INCLUDE_DIRS}
	${G2O_INCLUDE_DIRS}
	${CSparse_INCLUDE_DIRS}
	${EIGEN3_INCLUDE_DIRS}
	"/usr/include/suitesparse/"
)



add_executable(testG2O src/main.cpp)
target_link_libraries(testG2O 
    ${OpenCV_LIBRARIES} 
    ${CSPARSE_LIBRARY}
    ${CHOLMOD_LIBRARIES}
    g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension
    )