//
// Created by leo on 2019/7/30.
//
#ifndef TCP_SERVER_TCP_SERVER_H
#define TCP_SERVER_TCP_SERVER_H

#include <cstring>    // sizeof()
#include <iostream>
#include <string>
#include <sstream>  // stringstream
// headers for socket(), getaddrinfo() and friends
#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>    // close()
// #include <algorithm>  // remove_if()

#define RECEIVE_LENGTH 30
#define LOCAL_PORT "9090"

#define MAX_SPEED 255
#define adjust_scale 1
#define half_scale 1

//                                  环 灯    前后  左右   转向  上下
#define SEND_LIGHTS_ON              1, 1,    0,    0,    0,    0
#define SEND_FORWARD                1, 0,   99,    0,    1,    0
#define SEND_BACKWARD               1, 0, -100,    0,    1,    0
#define SEND_LEFT                   1, 0,    0,   99,    1,    0
#define SEND_RIGHT                  1, 0,    0, -100,    1,    0
#define SEND_TURN_LEFT              1, 0,    0,    0,   99,    0
#define SEND_TURN_RIGHT             1, 0,    0,    0, -100,    0
#define SEND_UP                     0, 0,    0,    0,    1,   99
#define SEND_DOWN                   0, 0,    0,    0,    1, -100
#define SEND_HALF_FORWARD           1, 0,   40 * half_scale,    0 * half_scale,    1,    0
#define SEND_HALF_BACKWARD          1, 0,  -40 * half_scale,    0 * half_scale,    1,    0
#define SEND_HALF_LEFT              1, 0,    0 * half_scale,   40 * half_scale,    1,    0
#define SEND_HALF_RIGHT             1, 0,    0 * half_scale,  -40 * half_scale,    1,    0
#define SEND_HALF_TURN_LEFT         1, 0,    0,    0,   30,    0
#define SEND_HALF_TURN_RIGHT        1, 0,    0,    0,  -30,    0
#define SEND_HALF_UP                0, 0,    0,    0,    1,   50
#define SEND_HALF_DOWN              0, 0,    0,    0,    1,  -50
#define SEND_SLEEP                  1, 0,    0,    0,    1,    0
#define SEND_DIVE_FORWARD           1, 0,   37,    0,    1, -100
#define SEND_DIVE_BACKWARD          1, 0,  -37,    0,    1, -100
#define SEND_DIVE_LEFT              1, 0,    0,   37,    1, -100
#define SEND_DIVE_RIGHT             1, 0,    0,  -37,    1, -100
#define SEND_DIVE_TURN_LEFT         1, 0,    0,    0,   37, -100
#define SEND_DIVE_TURN_RIGHT        1, 0,    0,    0,  -37, -100
#define SEND_ADJUST_FORWARD         1, 0,   33 * adjust_scale,    0 * adjust_scale,    1,    0
#define SEND_ADJUST_BACKWARD        1, 0,  -33 * adjust_scale,    0 * adjust_scale,    1,    0
#define SEND_ADJUST_LEFT            1, 0,    0 * adjust_scale,   37 * adjust_scale,    1,    0
#define SEND_ADJUST_RIGHT           1, 0,    0 * adjust_scale,  -33 * adjust_scale,    1,    0
#define SEND_ADJUST_TURN_LEFT       1, 0,    0,    0,   25,    0
#define SEND_ADJUST_TURN_RIGHT      1, 0,    0,    0,  -25,    0
#define SEND_DIVE_ADJUST_FORWARD    1,  0,   25,   0,    1,  -50
#define SEND_DIVE_ADJUST_BACKWARD   1,  0,  -25,   0,    1,  -50
#define SEND_DIVE_ADJUST_LEFT       1,  0,    0,  37,    1,  -50
#define SEND_DIVE_ADJUST_RIGHT      1,  0,    0, -25,    1,  -50


class TCP_Server {
public:
    int isOneLeak = 0;
    int isTwoLeak = 0;
    float depth = 0;
    float adjust_rate = 1.0;  // density_rate = sea_water_density / standard_water_density
    int land_count = 0;
    int count_thresh = 20;  // the unit is almost s, often smaller than 1s
    float pre_depth = 0.0;
    float depth_diff_thresh = 3.0;  // unit is cm
    float depth_diff = 100.0;

    TCP_Server();
    ~TCP_Server();
    void recvMsg();
    void sendMsg(bool is_close_loop, bool is_lights_on, int front_back, int left_right, int course, int up_down);
    bool is_landed(bool land_flag);
private:
    int is_new = 0;
    int sockFD;
    int newFD;
    char receive[RECEIVE_LENGTH];
    addrinfo *res;
    sockaddr_storage client_addr;
    socklen_t client_addr_size = sizeof(client_addr);
};

void run_rov();
#endif //TCP_SERVER_TCP_SERVER_H
