//
// Created by SONGZhuHeng on 2019/7/30.
//

#include "rov.h"
#include "utils.h"
#include "color.h"

#include <random>

TCP_Server::TCP_Server(void)
// initial the socket
{
    auto &portNum = LOCAL_PORT;  // set local port number to 9090
    const unsigned int backLog =
        8;  // number of connections allowed on the incoming queue

    // addrinfo hints, *res, *p;
    addrinfo hints,
        *p;  // we need 2 pointers, res to hold and p to iterate over
    memset(&hints, 0, sizeof(hints));

    // for more explanation, man socket
    hints.ai_family = AF_UNSPEC;  // don't specify which IP version to use yet
    hints.ai_socktype =
        SOCK_STREAM;  // SOCK_STREAM refers to TCP, SOCK_DGRAM will be?
    hints.ai_flags = AI_PASSIVE;

    // man getaddrinfo
    int gAddRes = getaddrinfo(NULL, portNum, &hints, &res);
    if (gAddRes != 0) {
        std::cerr << gai_strerror(gAddRes) << "\n";
    }

    // std::cout << "Detecting addresses" << std::endl;

    unsigned int numOfAddr = 0;
    char ipStr[INET6_ADDRSTRLEN];  // ipv6 length makes sure both ipv4/6
                                   // addresses can be stored in this variable

    // Now since getaddrinfo() has given us a list of addresses
    // we're going to iterate over them and ask user to choose one
    // address for program to bind to
    for (p = res; p != NULL; p = p->ai_next) {
        void *addr;
        std::string ipVer;

        // if address is ipv4 address
        if (p->ai_family == AF_INET) {
            ipVer = "IPv4";
            sockaddr_in *ipv4 = reinterpret_cast<sockaddr_in *>(p->ai_addr);
            addr = &(ipv4->sin_addr);
            ++numOfAddr;
        }

        // if address is ipv6 address
        else {
            ipVer = "IPv6";
            sockaddr_in6 *ipv6 = reinterpret_cast<sockaddr_in6 *>(p->ai_addr);
            addr = &(ipv6->sin6_addr);
            ++numOfAddr;
        }

        // convert IPv4 and IPv6 addresses from binary to text form
        inet_ntop(p->ai_family, addr, ipStr, sizeof(ipStr));
        // std::cout << "(" << numOfAddr << ") " << ipVer << " : " << ipStr <<
        // std::endl;
    }

    // if no addresses found :(
    if (!numOfAddr) {
        std::cerr << "Found no host address to use\n";
    }

    // choose IPv4 address
    unsigned int choice = 0;
    bool madeChoice = false;
    choice = 1;
    madeChoice = true;

    p = res;

    sockFD = socket(p->ai_family, p->ai_socktype, p->ai_protocol);
    // only for situation that the port restarted right away while sockets are
    // still active on its port
    int opt = 1;
    setsockopt(sockFD, SOL_SOCKET, SO_REUSEADDR, (const void *)&opt,
               sizeof(opt));
    if (sockFD == -1) {
        std::cerr << "Error while creating socket\n";
        freeaddrinfo(res);
    }

    int bindR = bind(sockFD, p->ai_addr, p->ai_addrlen);
    if (bindR == -1) {
        std::cerr << "Error while binding socket\n";
        // if some error occurs, make sure to close socket and free resources
        close(sockFD);
        freeaddrinfo(res);
    }

    // finally start listening for connections on our socket
    int listenR = listen(sockFD, backLog);
    if (listenR == -1) {
        std::cerr << "Error while Listening on socket\n";
        // if some error occurs, make sure to close socket and free resources
        close(sockFD);
        freeaddrinfo(res);
    }
}

TCP_Server::~TCP_Server(void)
// release the socket
{
    close(newFD);
    close(sockFD);
    freeaddrinfo(res);
}

void TCP_Server::recvMsg(void)
// receive message from the ROV
{
    if (!is_new) {
        newFD = accept(sockFD, (sockaddr *)&client_addr, &client_addr_size);
        is_new = 1;
    }
    if (newFD == -1) {
        std::cerr << "Error while Accepting on socket\n";
        return;
    }
    memset(receive, 0, sizeof(receive));
    auto bytes_recv = recv(newFD, receive, 27, 0);

    if (receive[4] == '\xaa') {
        isOneLeak = 1;
    }
    if (receive[7] == '\xaa') {
        isTwoLeak = 1;
    }
    depth = (float)((int)(receive[8]) * 256 + (int)(receive[9]));  // the unit is cm
    // 此处adjust_rate即README中修正参数k
    depth = depth / adjust_rate;
    // std::cout << isOneLeak << std::endl;
    // std::cout << isTwoLeak << std::endl;
    // std::cout << depth << std::endl;
}

void TCP_Server::sendMsg(bool is_close_loop, bool is_lights_on, int front_back, int left_right, int course, int up_down)
// 是否闭环, 是否开灯, 前进后退, 左右平移, 航向角, 上升下潜
// 速度值为-100到100的整数, 表示百分值
{
    std::string response = "\xfe\xfe";
    std::string response_inter;
    if (is_lights_on)
    {
        response.assign("\xfe\xfe\x01\x0f\x01\xf4\x01\xf4\x05\xdc\x05\xdc\x00\x00\x00\x00\x00\x00\x00\x00\x7f\x7f\x7f\x7f\x00\xfd\xfd", 27);
    }
    else
    {
        if (is_close_loop)
        {
            // 信息字
            response_inter.assign("\x03\x00", 2);
            response = response + response_inter;
        }
        else
        {
            // 信息字
            response_inter.assign("\x01\x00", 2);
            response = response + response_inter;
        }
        response_inter.assign("\x03\xb6\x03\xb6\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00", 16);
        response = response + response_inter;
        int value;
        // 前后
        value = 127 - front_back * 128 / 100;
        char a = (char)value;
        response = response + (char)(value);
        // 侧移
        response = response + (char)(127 - left_right * 128 / 100);
        // 航向角
        response = response + (char)(127 - course * 128 / 100);
        // 上升下潜
        response = response + (char)(127 - up_down * 128 / 100);
        // 生成BCC(Block Check Character/信息组校验码, 也称异或校验码)
        response_inter.assign("\x00", 1);
        response = response + response_inter;
        for (int i = 0; i < 24; i++)
        {
            response[24] = response[24] ^ response[i];
        }
        // 帧尾
        response = response + "\xfd\xfd";
    }
    auto bytes_sent = send(newFD, response.data(), response.length(), 0);
}

extern float max_depth, curr_depth;
bool TCP_Server::is_landed(bool land){ // 结合上一时刻是否位于海底和深度波动判断当前是否位于海底
    this->depth_diff = this->depth - this->pre_depth;
    this->pre_depth = this->depth;
    if (land) { // 更新海底深度

    }
    if (this->depth_diff < this->depth_diff_thresh) { // 深度持续稳定时间计时
        this->land_count++;
    }  else { // 当深度变化幅度超过阈值时判定为不在海底并归零深度持续稳定时间
        this->land_count = 0;
        return false;
    }
    if (this->land_count >= this->count_thresh) { // land_count超过阈值count_thresh时判定为坐底, 当land_count和count_thresh过小时会产生噪声
//        print(BOLDGREEN, "ROV: landed, update max depth " << depth);
        max_depth = this->depth;
        return true;
    }
    return false;
}

TCP_Server server;
extern bool run_rov_flag;
extern int rov_key, send_byte;
extern bool rov_half_speed, land, manual_stop, grasping_done;
extern std::vector<int> target_loc;
extern cv::Size vis_size;
extern bool second_dive;


void run_rov() {
    std::random_device rd;
//    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 5);
    int cruise_state = 0; // 巡航状态
    float width_thresh = 0.2; // 微调ROI阈值
    float height_thresh = 0.3;
    float drift_width = 0.0; // 目标漂移值
    float drift_height = 0.0;
    bool dive_ready = false;
    float x_ref = 0.5;
    float y_ref = 0.5;
    int send_times = 0;
    if (run_rov_flag > 0) { // 确认上位机与ROV通信建立成功并开灯
        print(BOLDGREEN, "ROV: try to first receive");
        server.recvMsg();
        print(BOLDGREEN, "ROV: first receive done, current depth: " << server.depth);
//        server.sendMsg(SEND_LIGHTS_ON);
    }
    while (run_rov_flag) { // 进入ROV动作控制循环, 整个比赛过程中都应当处于这个循环中
        server.recvMsg();
        if (server.depth > 0) curr_depth = server.depth;
        switch (rov_key) { // 键盘键值与ROV动作映射
            case 99:  // c
                server.sendMsg(SEND_SLEEP);
                break;
            case 105:  // k
                if (rov_half_speed) server.sendMsg(SEND_HALF_FORWARD);
                else server.sendMsg(SEND_FORWARD);
                break;
            case 106:  // j
                if (rov_half_speed) server.sendMsg(SEND_HALF_TURN_LEFT);
                else server.sendMsg(SEND_TURN_LEFT);
                break;
            case 107:  // k
                if (rov_half_speed)  server.sendMsg(SEND_HALF_BACKWARD);
                else server.sendMsg(SEND_BACKWARD);
                break;
            case 108:  // l
                if (rov_half_speed) server.sendMsg(SEND_HALF_TURN_RIGHT);
                else  server.sendMsg(SEND_TURN_RIGHT);
                break;
            case 44:  // ,
                server.sendMsg(SEND_DOWN);
                break;
            case 46:  // .
                if (rov_half_speed) server.sendMsg(SEND_HALF_UP);
                else server.sendMsg(SEND_UP);
                break;
            case 59:  // ; 坐底. 从这一步开始为自主控制.
                print(BOLDGREEN, "ROV: diving !!!");
                grasping_done = false;
                second_dive = false;
                while (!manual_stop && !grasping_done && !second_dive) { // 当未人为操作且软体臂抓取未完成时持续坐底
                    server.sendMsg(SEND_DOWN);
                    server.recvMsg();
                    if (server.depth > 0) land = server.is_landed(land); // 判定是否到达海底
                }
                server.land_count = 0;
                land = false;  // 结束坐底
                if (manual_stop) rov_key = 99;
                else rov_key = 39; // 开始上浮并定深
                break;
            case 39:  // '  定深, 全速上浮3s, 悬停2s等ROV静止后获取当前深度
                print(BOLDBLUE, "ROV: try to stably floating, second_dive? " << second_dive);
                if (second_dive) {
                    for (unsigned char i = 0; i < 10; i++) server.sendMsg(SEND_SLEEP);
                    delay(2);
                } else {
                    print(BOLDBLUE, "ROV: UP");
                    for (unsigned char i = 0; i < 10; i++) server.sendMsg(SEND_UP);
                    delay(4);
                    for (unsigned char i = 0; i < 10; i++) server.sendMsg(SEND_SLEEP);
                    delay(2);
                }
                while (true) {
                    delay(1);
                    for (unsigned char i = 0; i < 10; i++) server.sendMsg(SEND_UP);
                    server.recvMsg();
                    if (server.depth > 0) {
                        print(BOLDBLUE, "ROV: floating at " << server.depth);
                        break;
                    }
                }
                if (manual_stop) rov_key = 99;
                else rov_key = 47;
                break;
            case 47:  // /  视野内无目标时遍历水域
                print(BOLDMAGENTA, "ROV: cruising");
                while ((!manual_stop)) {
                    delay(1);
                    if (target_loc.at(2) != 0 && target_loc.at(3) != 0) { // target_loc.at 2, 3位为目标的width, height
                        rov_key = 61;
                        break;
                    }
                    server.sendMsg(SEND_SLEEP);
                    // 没有target就蛇形走位, 将状态分为f2r (forward to right), r2f, f2l, f2f四种, (首开尾闭), 依次进行 cruise_state依次为0, 1, 2, 3
//                    if (cruise_state < 3) ++cruise_state; // 循环cruise_state
//                    else cruise_state = 0;
//                    switch (cruise_state) { // cruise_state与动作映射
//                        case 0:  // f2r
//                            server.sendMsg(SEND_HALF_FORWARD);
//                            delay_ms(3000);  // 1.5s
//                            server.sendMsg(SEND_HALF_TURN_RIGHT);
//                            delay_ms(1000);  // FIXME
//                            break;
//                        case 1:  // r2f
//                            server.sendMsg(SEND_HALF_FORWARD);
//                            delay_ms(3000);  // 3s
//                            server.sendMsg(SEND_HALF_TURN_LEFT);
//                            delay_ms(1000);  // FIXME
//                            break;
//                        case 2:  // f2l
//                            server.sendMsg(SEND_HALF_FORWARD);
//                            delay_ms(3000);
//                            server.sendMsg(SEND_HALF_TURN_LEFT);
//                            delay_ms(1000);  // FIXME
//                            break;
//                        case 3:  // l2f
//                            server.sendMsg(SEND_HALF_FORWARD);
//                            delay_ms(3000);
//                            server.sendMsg(SEND_TURN_RIGHT);
//                            delay_ms(1000);  // FIXME
//                            break;
//                        default:
//                            break;
//                    }
                }
                break;
            case 61:  // = 实时微调水平位置并全速下潜
                print(BOLDMAGENTA, "ROV: aming");
                send_times = 0;
                while ((!manual_stop)) {
                    delay(1);  // 0.1s FIXME: delay may too long
                    send_times++;
                    if (target_loc.at(2) == 0 || target_loc.at(3) == 0) {  // 当目标丢失时跳出循环到case59 坐底
                        dive_ready = false;
                        break;
                    }
                    if (second_dive) {
                        y_ref = 0.6;
                        height_thresh = 0.2;
                    } else {
                        y_ref = 0.4;
                        height_thresh = 0.3;
                    }
                    if (target_loc.at(0) < (float)vis_size.width * (x_ref - width_thresh / 2) || // 先判定是否有左右漂移及漂移值, 向左为正
                        target_loc.at(0) > (float)vis_size.width * (x_ref + width_thresh / 2))   // 要注意图像坐标系原点在图像左上角
                        drift_width = vis_size.width * x_ref - target_loc.at(0);
                    else drift_width = 0;
                    if (target_loc.at(1) < (float)vis_size.height * (y_ref - height_thresh / 2) || // 然后判断是否有前后漂移及漂移值, 向上为正
                        target_loc.at(1) > (float)vis_size.height * (y_ref + height_thresh / 2))
                        drift_height = vis_size.height * y_ref - target_loc.at(1);
                    else
                        drift_height = 0;
                    if (drift_width == 0 && drift_height == 0) { // 当目标在ROI内时全速下潜
                        dive_ready = true;
                        break;
                    } else if (std::abs(drift_width) >= std::abs(drift_height)) { // 比较左右漂移值和前后漂移值大小, 优先微调漂移更严重方向
                        if (drift_width > 0) { // 目标在视野中偏左则左转, 偏右则右转, 注意左右偏移时用转动来微调
//                            if (send_times % 4 == 0){
//                                server.sendMsg(SEND_DIVE_ADJUST_LEFT);
//                                print(BOLDMAGENTA, "ROV: SEND_DIVE_LEFT");
//                            } else {
;                           server.sendMsg(SEND_ADJUST_LEFT);
                            print(BOLDMAGENTA, "ROV: SEND_ADJUST_LEFT");
//                            }
                        } else {
//                            if (send_times % 4 == 0) {
//                                server.sendMsg(SEND_DIVE_ADJUST_RIGHT);
//                                print(BOLDMAGENTA, "ROV: SEND_DIVE_RIGHT");
//                            } else {
                            server.sendMsg(SEND_ADJUST_RIGHT);
                            print(BOLDMAGENTA, "ROV: SEND_ADJUST_RIGHT");
//                            }
                        }
                    } else { // 偏前则前移, 偏后则后移
                        if (drift_height > 0) {
//                            if (send_times % 4 == 0) {
//                                server.sendMsg(SEND_DIVE_ADJUST_FORWARD);
//                                print(BOLDMAGENTA, "ROV: SEND_DIVE_FORWARD");
//                            } else {
                            server.sendMsg(SEND_ADJUST_FORWARD);
                            print(BOLDMAGENTA, "ROV: SEND_ADJUST_FORWARD");
//                            }
                        } else {
//                            if (send_times % 4 == 0) {
//                                server.sendMsg(SEND_DIVE_ADJUST_BACKWARD);
//                                print(BOLDMAGENTA, "ROV: SEND_DIVE_BACKWARD");
//                            } else {
                            server.sendMsg(SEND_ADJUST_BACKWARD);
                            print(BOLDMAGENTA, "ROV: SEND_ADJUST_BACKWARD");
//                            }
                        }
                    }
                }
                if (manual_stop) rov_key = 99;
                else if(dive_ready) {
                    rov_key = 59;  // 跳回坐底
                    dive_ready = false;
                }
                else rov_key = 47;
                break;
            default:
                server.sendMsg(SEND_SLEEP);
                break;
        }
    }
    for (unsigned char i=0; i<10; i++)
        server.sendMsg(SEND_SLEEP);
    print(RED, "QUIT: run_rov quit");
}
