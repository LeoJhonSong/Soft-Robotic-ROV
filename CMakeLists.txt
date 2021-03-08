cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
#set(CMAKE_CXX_STANDARD 14)
#set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
project(Soft-Robotics-ROV)

set(CMAKE_PREFIX_PATH "~/local/libtorch")
find_package(Torch REQUIRED)
find_package( OpenCV 3 REQUIRED)
#搜索ArUco依赖库
find_package(aruco REQUIRED)
set(OpenCV_LIBS ${OpenCV_LIBS})
include_directories( ${OpenCV_INCLUDE_DIRS} include)
include_directories( ${aruco_INCLUDE_DIRS})

file(GLOB SOURCES "src/*.cpp")
add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES} ${OpenCV_LIBS} ${aruco_LIBS})
