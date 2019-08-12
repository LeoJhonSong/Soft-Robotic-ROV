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

//                               *       *       *       *       *       *       *       *       *       *       *   *   *   *   *   *
//                               帧头    信息字  LED1    LED2    舵机1   舵机2   舵预1   舵预2   舵预3   舵预3   前后侧移方向上下校验帧尾
#define LIGHTS_ON               "\xfe\xfe\x01\x0f\x01\xf4\x01\xf4\x05\xdc\x05\xdc\x00\x00\x00\x00\x00\x00\x00\x00\x7f\x7f\x7f\x7f\x00\xfd\xfd"

#define MOVE_FORWARD            "\xfe\xfe\x03\x00\x03\xb6\x03\xb6\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x7f\x7f\x7f\x7c\xfd\xfd"
#define MOVE_BACKWARD           "\xfe\xfe\x03\x00\x03\xb6\x03\xb6\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\x7f\x7f\x7f\x83\xfd\xfd"
#define MOVE_LEFT               "\xfe\xfe\x03\x00\x03\xb6\x03\xb6\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x7f\x00\x7f\x7f\x7c\xfd\xfd"
#define MOVE_RIGHT              "\xfe\xfe\x03\x00\x03\xb6\x03\xb6\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x7f\xff\x7f\x7f\x83\xfd\xfd"
#define MOVE_TURN_LEFT          "\xfe\xfe\x03\x00\x03\xb6\x03\xb6\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x7f\x7f\x00\x7f\x7c\xfd\xfd"
#define MOVE_TURN_RIGHT         "\xfe\xfe\x03\x00\x03\xb6\x03\xb6\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x7f\x7f\xff\x7f\x83\xfd\xfd"
#define MOVE_UP                 "\xfe\xfe\x03\x00\x03\xb6\x03\xb6\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x7f\x7f\x7f\x00\x7c\xfd\xfd"
#define MOVE_DOWN               "\xfe\xfe\x03\x00\x03\xb6\x03\xb6\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x7f\x7f\x7f\xff\x83\xfd\xfd"
#define MOVE_HALF_FORWARD       "\xfe\xfe\x03\x00\x03\xb6\x03\xb6\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x3f\x7f\x7f\x7f\x7c\xfd\xfd"
#define MOVE_HALF_BACKWARD      "\xfe\xfe\x03\x00\x03\xb6\x03\xb6\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xbf\x7f\x7f\x7f\x83\xfd\xfd"
#define MOVE_HALF_LEFT          "\xfe\xfe\x03\x00\x03\xb6\x03\xb6\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x7f\x3f\x7f\x7f\x7c\xfd\xfd"
#define MOVE_HALF_RIGHT         "\xfe\xfe\x03\x00\x03\xb6\x03\xb6\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x7f\xbf\x7f\x7f\x83\xfd\xfd"
#define MOVE_HALF_TURN_LEFT     "\xfe\xfe\x03\x00\x03\xb6\x03\xb6\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x7f\x7f\x3f\x7f\x7c\xfd\xfd"
#define MOVE_HALF_TURN_RIGHT    "\xfe\xfe\x03\x00\x03\xb6\x03\xb6\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x7f\x7f\xbf\x7f\x83\xfd\xfd"
#define MOVE_HALF_UP            "\xfe\xfe\x03\x00\x03\xb6\x03\xb6\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x7f\x7f\x7f\x3f\x7c\xfd\xfd"
#define MOVE_HALF_DOWN          "\xfe\xfe\x03\x00\x03\xb6\x03\xb6\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x7f\x7f\x7f\xbf\x83\xfd\xfd"
#define MOVE_SLEEP              "\xfe\xfe\x03\x00\x03\xb6\x03\xb6\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x7f\x7f\x7f\x7f\x03\xfd\xfd"

//水平方向四分之一速微调并且全速向下
#define MOVE_ADJUST_FORWARD     "\xfe\xfe\x03\x00\x03\xb6\x03\xb6\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x5f\x7f\x7f\xff\x23\xfd\xfd"
#define MOVE_ADJUST_BACKWARD    "\xfe\xfe\x03\x00\x03\xb6\x03\xb6\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x9f\x7f\x7f\xff\xe3\xfd\xfd"
#define MOVE_ADJUST_LEFT        "\xfe\xfe\x03\x00\x03\xb6\x03\xb6\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x7f\x5f\x7f\xff\x23\xfd\xfd"
#define MOVE_ADJUST_RIGHT       "\xfe\xfe\x03\x00\x03\xb6\x03\xb6\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x7f\x9f\x7f\xff\xe3\xfd\xfd"
#define MOVE_ADJUST_TURN_LEFT   "\xfe\xfe\x03\x00\x03\xb6\x03\xb6\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x7f\x5f\x7f\xff\x23\xfd\xfd"
#define MOVE_ADJUST_TURN_RIGHT  "\xfe\xfe\x03\x00\x03\xb6\x03\xb6\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x7f\x9f\x7f\xff\xe3\xfd\xfd"

// 仅水平微调
// #define MOVE_ADJUST_FORWARD     "\xfe\xfe\x03\x00\x03\xb6\x03\xb6\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x5f\x7f\x7f\x7f\x23\xfd\xfd"
// #define MOVE_ADJUST_BACKWARD    "\xfe\xfe\x03\x00\x03\xb6\x03\xb6\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x9f\x7f\x7f\x7f\xe3\xfd\xfd"
// #define MOVE_ADJUST_LEFT        "\xfe\xfe\x03\x00\x03\xb6\x03\xb6\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x7f\x5f\x7f\x7f\x23\xfd\xfd"
// #define MOVE_ADJUST_RIGHT       "\xfe\xfe\x03\x00\x03\xb6\x03\xb6\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x7f\x9f\x7f\x7f\xe3\xfd\xfd"
// #define MOVE_ADJUST_TURN_LEFT   "\xfe\xfe\x03\x00\x03\xb6\x03\xb6\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x7f\x5f\x7f\x7f\x23\xfd\xfd"
// #define MOVE_ADJUST_TURN_RIGHT  "\xfe\xfe\x03\x00\x03\xb6\x03\xb6\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x7f\x9f\x7f\x7f\xe3\xfd\xfd"

#define SEND_LIGHTS_ON          0
#define SEND_FORWARD            1
#define SEND_BACKWARD           2
#define SEND_LEFT               3
#define SEND_RIGHT              4
#define SEND_TURN_LEFT          5
#define SEND_TURN_RIGHT         6
#define SEND_UP                 7
#define SEND_DOWN               8
#define SEND_HALF_FORWARD       9
#define SEND_HALF_BACKWARD      10
#define SEND_HALF_LEFT          11
#define SEND_HALF_RIGHT         12
#define SEND_HALF_TURN_LEFT     13
#define SEND_HALF_TURN_RIGHT    14
#define SEND_HALF_UP            15
#define SEND_HALF_DOWN          16
#define SEND_SLEEP              17
#define SEND_ADJUST_FORWARD     18
#define SEND_ADJUST_BACKWARD    19
#define SEND_ADJUST_LEFT        20
#define SEND_ADJUST_RIGHT       21
#define SEND_ADJUST_TURN_LEFT   22
#define SEND_ADJUST_TURN_RIGHT  23

class TCP_Server {
    public:
        int isOneLeak = 0;
        int isTwoLeak = 0;
        float depth = 0;
        float adjust_rate = 1.0;  // density_rate = sea_water_density / standard_water_density
        int land_count = 0;
        int count_thresh = 15;  // the unit is almost s, often smaller than 1s
        float pre_depth = 0.0;
        float depth_diff_thresh = 6.0;  // unit is cm
        float depth_diff = 100.0;
        float max_depth = 0.0;


        TCP_Server();
        ~TCP_Server();
        void recvMsg();
        void sendMsg(int move);
        bool is_landed();
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
