cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
#set(CMAKE_CXX_STANDARD 14)
#set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
project(Soft-Robotics-ROV)

set(OpenCV_LIBS ${OpenCV_LIBS})
# for desktop
set(CMAKE_PREFIX_PATH "~/local/libtorch")
# for jetson
set(CMAKE_PREFIX_PATH "~/.local/lib/python3.6/site-packages/torch")

file(GLOB SOURCES "src/*.cpp")
add_executable(${PROJECT_NAME} ${SOURCES})

find_package(Torch REQUIRED)
find_package( OpenCV 3 REQUIRED)
find_package(aruco REQUIRED)
# set path to find *.h files
include_directories(include)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${aruco_INCLUDE_DIRS})
# set path to find *.so, *.a fils
link_directories(/usr/local/lib)

target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES} ${OpenCV_LIBS} ${aruco_LIBS})