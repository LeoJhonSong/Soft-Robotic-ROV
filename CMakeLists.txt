cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
project(ResDet)

find_package(Torch REQUIRED)
set(OpenCV_DIR "~/local/opencv3.4.cuda10.0/share/OpenCV")
find_package( OpenCV 3 REQUIRED )
set(OpenCV_LIBS ${OpenCV_LIBS})
find_package (gflags REQUIRED)
include_directories (${gflags_INCLUDE_DIR})

### test
message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
###
include_directories( ${OpenCV_INCLUDE_DIRS} include)
file(GLOB SOURCES "src/*.cpp")
add_executable(ResDet ${SOURCES})
target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES} ${OpenCV_LIBS} gflags glog)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 11)
# /home/sean/local/lib/libglog.so
