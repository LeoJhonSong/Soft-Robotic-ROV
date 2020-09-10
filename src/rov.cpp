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
    auto &portNum = LOCAL_PORT;     // set local port number to 9090
    const unsigned int backLog = 8; // number of connections allowed on the incoming queue

    // addrinfo hints, *res, *p;
    addrinfo hints, *p; // we need 2 pointers, res to hold and p to iterate over
    memset(&hints, 0, sizeof(hints));

    // for more explanation, man socket
    hints.ai_family = AF_UNSPEC;     // don't specify which IP version to use yet
    hints.ai_socktype = SOCK_STREAM; // SOCK_STREAM refers to TCP, SOCK_DGRAM will be?
    hints.ai_flags = AI_PASSIVE;

    // man getaddrinfo
    int gAddRes = getaddrinfo(NULL, portNum, &hints, &res);
    if (gAddRes != 0)
    {
        std::cerr << gai_strerror(gAddRes) << "\n";
    }

    // std::cout << "Detecting addresses" << std::endl;

    unsigned int numOfAddr = 0;
    char ipStr[INET6_ADDRSTRLEN]; // ipv6 length makes sure both ipv4/6
    // addresses can be stored in this variable

    // Now since getaddrinfo() has given us a list of addresses
    // we're going to iterate over them and ask user to choose one
    // address for program to bind to
    for (p = res; p != NULL; p = p->ai_next)
    {
        void *addr;
        std::string ipVer;

        // if address is ipv4 address
        if (p->ai_family == AF_INET)
        {
            ipVer = "IPv4";
            sockaddr_in *ipv4 = reinterpret_cast<sockaddr_in *>(p->ai_addr);
            addr = &(ipv4->sin_addr);
            ++numOfAddr;
        }

        // if address is ipv6 address
        else
        {
            ipVer = "IPv6";
            sockaddr_in6 *ipv6 = reinterpret_cast<sockaddr_in6 *>(p->ai_addr);
            addr = &(ipv6->sin6_addr);
            ++numOfAddr;
        }

        // convert IPv4 and IPv6 addresses from binary to text form
        inet_ntop(p->ai_family, addr, ipStr, sizeof(ipStr));
        // std::cout << "(" << numOfAddr << ") " << ipVer << " : " << ipStr << std::endl;
    }

    // if no addresses found :(
    if (!numOfAddr)
    {
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
    if (sockFD == -1)
    {
        std::cerr << "Error while creating socket\n";
        freeaddrinfo(res);
    }

    int bindR = bind(sockFD, p->ai_addr, p->ai_addrlen);
    if (bindR == -1)
    {
        std::cerr << "Error while binding socket\n";
        // if some error occurs, make sure to close socket and free resources
        close(sockFD);
        freeaddrinfo(res);
    }

    // finally start listening for connections on our socket
    int listenR = listen(sockFD, backLog);
    if (listenR == -1)
    {
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
    if (!is_new)
    {
        newFD = accept(sockFD, (sockaddr *)&client_addr, &client_addr_size);
        is_new = 1;
    }
    if (newFD == -1)
    {
        std::cerr << "Error while Accepting on socket\n";
        return;
    }
    memset(receive, 0, sizeof(receive));
    auto bytes_recv = recv(newFD, receive, 27, 0);

    if (receive[4] == '\xaa')
    {
        isOneLeak = 1;
    }
    if (receive[7] == '\xaa')
    {
        isTwoLeak = 1;
    }
    depth = (float)((int)(receive[8]) * 256 + (int)(receive[9])); // the unit is cm
    // 此处adjust_rate即README中修正参数k
    depth = depth / adjust_rate;
    // std::cout << isOneLeak << std::endl;
    // std::cout << isTwoLeak << std::endl;
    // std::cout << depth << std::endl;
}

void TCP_Server::sendMsg(bool is_close_loop, int is_lights_on, int front_back, int left_right, int course, int up_down)
// 是否闭环, 是否开灯, 前进后退, 左右平移, 航向角, 上升下潜
// 速度值为-100到100的整数, 表示百分值
{
    std::string response = "\xfe\xfe";
    std::string response_inter;
    if (is_lights_on == 1)
    {
        response.assign("\xfe\xfe\x01\x0f\x01\xf4\x01\xf4\x05\xdc\x05\xdc\x00\x00\x00\x00\x00\x00\x00\x00\x7f\x7f\x7f\x7f\x00\xfd\xfd", 27);
    }
    else if (is_lights_on == 2)
    {
        response.assign("\xfe\xfe\x01\x0f\x00\x00\x00\x00\x05\xdc\x05\xdc\x00\x00\x00\x00\x00\x00\x00\x00\x7f\x7f\x7f\x7f\x0e\xfd\xfd", 27);
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
        if (value > MAX_SPEED)
        {
            value = MAX_SPEED;
        }
        else if (value < 0)
        {
            value = 0;
        }
        response = response + (char)(value);
        // 侧移
        value = 127 - left_right * 128 / 100;
        if (value > MAX_SPEED)
        {
            value = MAX_SPEED;
        }
        else if (value < 0)
        {
            value = 0;
        }
        response = response + (char)(value);
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
// 结合上一时刻是否位于海底和深度波动判断当前是否位于海底
bool TCP_Server::is_landed(bool land)
{
    this->depth_diff = this->depth - this->pre_depth;
    if (this->depth == 0)
    {
        return false;
    }
    this->pre_depth = this->depth;
    print(RED, "depth: " << this->pre_depth);
    if (land)
    { // 更新海底深度
    }
    if (this->depth_diff < this->depth_diff_thresh)
    { // 深度持续稳定时间计时
        this->land_count++;
    }
    else
    { // 当深度变化幅度超过阈值时判定为不在海底并归零深度持续稳定时间
        this->land_count = 0;
        return false;
    }
    if (this->land_count >= this->count_thresh)
    { // land_count超过阈值count_thresh时判定为坐底, 当land_count和count_thresh过小时会产生噪声
        if (std::abs(max_depth - this->depth) > 3)
        {
            print(BOLDGREEN, "ROV: landed, update max depth " << depth);
            max_depth = this->depth;
        }
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

void run_rov()
{
    // 巡航路线
    int side_sec = 3;
    int for_sec = 3;
    std::vector<int> cruise_second = {
            for_sec,
            for_sec +     side_sec,
        2 * for_sec +     side_sec,
        2 * for_sec + 2 * side_sec,
        3 * for_sec + 2 * side_sec,
        3 * for_sec + 3 * side_sec,
        4 * for_sec + 3 * side_sec,
        4 * for_sec + 4 * side_sec
    };
    float width_thresh = 0.2; // 微调ROI阈值
    float height_thresh = 0.3;
    float drift_width = 0.0; // 目标漂移值
    float drift_height = 0.0;
    bool dive_ready = false;
    bool second_dive_lost = false;
    float x_ref = 0.5;
    float y_ref = 0.5;

    time_t start, time_interval;
    unsigned char last_opt = 0;
    // 确认上位机与ROV通信建立成功并开灯
    if (run_rov_flag > 0)
    {
        print(BOLDGREEN, "ROV: try to first receive");
        server.recvMsg();
        print(BOLDGREEN, "ROV: first receive done, current depth: " << server.depth);
        server.sendMsg(SEND_LIGHTS_ON);
    }
    // 进入ROV动作控制循环, 整个比赛过程中都应当处于这个循环中
    int chances = 0;
    while (run_rov_flag)
    {
        server.recvMsg();
        if (server.depth > 0)
            curr_depth = server.depth;
        // 键盘键值与ROV动作映射
        switch (rov_key)
        {
        case 32: // 停止, space
            server.sendMsg(SEND_SLEEP);
            break;
        case 119: // 前进, w
            if (rov_half_speed)
                server.sendMsg(SEND_HALF_FORWARD);
            else
                server.sendMsg(SEND_FORWARD);
            break;
        case 65: // 左转, A (shift + a)
            if (rov_half_speed)
                server.sendMsg(SEND_HALF_TURN_LEFT);
            else
                server.sendMsg(SEND_TURN_LEFT);
            break;
        case 115: // 后退, s
            if (rov_half_speed)
                server.sendMsg(SEND_HALF_BACKWARD);
            else
                server.sendMsg(SEND_BACKWARD);
            break;
        case 68: // 右转, D (shift + d)
            if (rov_half_speed)
                server.sendMsg(SEND_HALF_TURN_RIGHT);
            else
                server.sendMsg(SEND_TURN_RIGHT);
            break;
        case 97: // 左侧移, a
            if (rov_half_speed)
                server.sendMsg(SEND_HALF_LEFT);
            else
                server.sendMsg(SEND_LEFT);
            break;
        case 100: // 右侧移, d
            if (rov_half_speed)
                server.sendMsg(SEND_HALF_RIGHT);
            else
                server.sendMsg(SEND_RIGHT);
            break;
        case 83: // 下潜, S (shift + s)
            if (rov_half_speed)
                server.sendMsg(SEND_HALF_DOWN);
            server.sendMsg(SEND_DOWN);
            break;
        case 87: // 上浮, W (shift + w)
            if (rov_half_speed)
                server.sendMsg(SEND_HALF_UP);
            else
                server.sendMsg(SEND_UP);
            break;
        case 108: // 关灯, l
            print(BOLDGREEN, "ROV: light off");
            server.sendMsg(SEND_LIGHTS_OFF);
            delay(1);
            rov_key = 32;
            break;
        case 76: // 开灯, L (shift + l)
            print(BOLDGREEN, "ROV: light on");
            server.sendMsg(SEND_LIGHTS_ON);
            delay(1);
            rov_key = 32;
            break;
        case 13: // 坐底, 回车 (\r) (从这一步开始为自主控制)
            print(BOLDGREEN, "ROV: diving !!!");
            grasping_done = false;
            second_dive = false;
            second_dive_lost = false;
            // 当未人为操作且软体臂抓取未完成时持续坐底
            while (!manual_stop && !grasping_done && !second_dive)
            {
                print(YELLOW, "hello");
                server.sendMsg(SEND_DOWN);
                server.recvMsg();
                if (server.depth > 0)
                    land = server.is_landed(land); // 判定是否到达海底
                print(RED, "land: " << land);
            }
            server.land_count = 0;
            land = false; // 结束坐底
            if (manual_stop)
                rov_key = 32;
            else
                rov_key = 117; // 开始上浮并定深
            break;
        case 117: // 上浮定深, u (全速上浮3s, 悬停2s等ROV静止后获取当前深度)
            print(BOLDBLUE, "ROV: try to stably floating, second_dive? " << second_dive << ", second_dive_lost? " << second_dive_lost);
            if (second_dive)
            {
                if (!second_dive_lost)
                {
                    server.sendMsg(SEND_SLEEP);
                    delay_ms(100);
                }
                else
                {
                    for (unsigned char i = 0; i < 1; i++)
                    {
                        server.sendMsg(SEND_UP);
                        delay(1);
                    }
                    for (unsigned char i = 0; i < 1; i++)
                    {
                        server.sendMsg(SEND_SLEEP);
                        delay(1);
                    }
                    second_dive = false;
                    second_dive_lost = false;
                }
            }
            else
            {
                for (unsigned char i = 0; i < 1; i++)
                {
                    server.sendMsg(SEND_UP);
                    delay(1);
                }
                for (unsigned char i = 0; i < 1; i++)
                {
                    server.sendMsg(SEND_SLEEP);
                    delay(1);
                }
            }
            while (true)
            {
                delay(1);
                for (unsigned char i = 0; i < 10; i++)
                    server.sendMsg(SEND_UP);
                server.recvMsg();
                if (server.depth > 0)
                {
                    print(BOLDBLUE, "ROV: floating at " << server.depth);
                    break;
                }
            }
            if (manual_stop)
                rov_key = 32;
            else if (second_dive)
                rov_key = 111;
            else
                rov_key = 99;
            break;
        case 99: // 视野内无目标时遍历水域, c
            print(BOLDMAGENTA, "ROV: cruising");
            last_opt = 0;
            start = time(nullptr);
            while (!manual_stop)
            {
                delay_ms(10);
                // 抓到目标后往前抖一下确保目标进框里
                if (grasping_done)
                {
                    print(BOLDMAGENTA, "ROV: grasping_done SEND_HALF_FORWARD for 2s before detection");
                    for (unsigned char i = 0; i < 2; i++)
                    {
                        server.sendMsg(SEND_FORWARD);
                        delay(1);
                    }
                    server.sendMsg(SEND_HALF_BACKWARD);
                    delay(1);
                    grasping_done = false;
                }
                time_interval = (time(nullptr) - start);
                // 发现目标后给一小段时间反向速度来减速, 减小水平速度对坐底的影响
                if (target_loc.at(2) != 0 && target_loc.at(3) != 0) // target_loc.at 2, 3位为目标的width, height
                {
                    if (last_opt == 1)
                    {
                        print(BOLDMAGENTA, "ROV: SEND_HALF_BACKWARD for 2s");
                        for (unsigned char i = 0; i < 2; i++)
                        {
                            server.sendMsg(SEND_HALF_BACKWARD);
                            delay(1);
                        }
                    }
                    else if (last_opt == 2)
                    {
                        print(BOLDMAGENTA, "ROV: SEND_HALF_LEFT for 2s");
                        for (unsigned char i = 0; i < 2; i++)
                        {
                            server.sendMsg(SEND_HALF_LEFT);
                            delay(1);
                        }
                    }
                    else if (last_opt == 3)
                    {
                        print(BOLDMAGENTA, "ROV: SEND_HALF_RIGHT for 2s");
                        for (unsigned char i = 0; i < 2; i++)
                        {
                            server.sendMsg(SEND_HALF_RIGHT);
                            delay(1);
                        }
                    }
                    if (last_opt > 0)
                        for (unsigned char i = 0; i < 2; i++)
                        {
                            server.sendMsg(SEND_SLEEP);
                            delay(1);
                        }
                    break;
                }
                // 未发现目标的巡航路线
                else if (time_interval <= cruise_second.at(0) ||
                         (time_interval > cruise_second.at(1) && time_interval <= cruise_second.at(2)) ||
                         (time_interval > cruise_second.at(3) && time_interval <= cruise_second.at(4)) ||
                         (time_interval > cruise_second.at(5) && time_interval <= cruise_second.at(6)))
                {
                    print(BOLDMAGENTA, "ROV: SEND_HALF_FORWARD");
                    last_opt = 1;
                    server.sendMsg(SEND_HALF_FORWARD);
                }
                else if ((time_interval > cruise_second.at(0) && time_interval <= cruise_second.at(1)) ||
                         (time_interval > cruise_second.at(6) && time_interval <= cruise_second.at(7)))
                {
                    print(BOLDMAGENTA, "ROV: SEND_HALF_RIGHT");
                    last_opt = 2;
                    server.sendMsg(SEND_HALF_RIGHT);
                }
                else if ((time_interval > cruise_second.at(2) && time_interval <= cruise_second.at(3)) ||
                         (time_interval > cruise_second.at(4) && time_interval <= cruise_second.at(5)))
                {
                    print(BOLDMAGENTA, "ROV: SEND_HALF_LEFT");
                    last_opt = 3;
                    server.sendMsg(SEND_HALF_LEFT);
                }
                else
                {
                    break;
                }
            }
            if (manual_stop)
                rov_key = 32;
            else if (time_interval > cruise_second.at(7))  // 运行一次巡航路线后坐底, 更新巡航高度
                rov_key = 13;
            else
                rov_key = 111;
            break;
        case 111: // 坐底至目标处, o
            print(BOLDYELLOW, "ROV: aiming");
            // FIXME 跳动超过x次, 放弃靠近目标, 转case13 (坐底)
            while ((!manual_stop) && chances < 3)
            {
                chances++;
                delay(1);  // 限制循环频率, 避免占用过高CPU
                // 目标丢失, 放弃靠近目标, 转case13 (坐底)
                if (target_loc.at(2) == 0 || target_loc.at(3) == 0)
                {
                    print(RED, "aaa");
                    rov_key = 13;
                    break;
                }
                // float target_x = std::abs(target_loc.at(0) / vis_size.width - 0.5);
                // float target_y = std::abs(target_loc.at(1) / vis_size.height - 1.0);
                float target_x = float(target_loc.at(0)) / vis_size.width - 0.5;
                print(RED, target_loc.at(0));
                float target_y = float(target_loc.at(1)) / vis_size.height - 0.8;
                // FIXME 如果目标在一个较大的阈值框外, 设置指向图像中心的水平满速; 如果目标在阈值框内, 水平速度置零. 同时全速坐底
                int speed_x = 0;
                int speed_y = 0;
                // if(std::abs(target_x) > 0.5 || std::abs(target_y) > 1.0)
                // {
                //     speed_x = target_x / std::min(target_x, target_y) *  100;
                //     speed_y = target_y / std::min(target_x, target_y) *  100;
                // }
                // if(target_x > 0.4)
                // {
                //     speed_x = target_x / std::min(target_x, target_y) *  100;
                //     speed_y = target_y / std::min(target_x, target_y) *  100;
                // }
                // while (std::abs(target_x) > 0.2)
                // {
                //     target_x = float(target_loc.at(0)) / vis_size.width - 0.5;
                //     speed_x = - (1/0.5) * target_x * 100;
                //     delay_ms(500);
                //     server.sendMsg(0, 0, speed_x, 0, 0, -100);
                //     print(RED, "target_x: " << target_x);
                // }
                
                // if (std::abs(target_x) > 0.2)
                // {
                //     speed_x = - (1/0.5) * target_x * 100;
                //     print(YELLOW, "speed_x: " << speed_x);
                // }
                if (std::abs(target_y) > 0.20)
                {
                    speed_y = -target_y * 100 * 0.6;
                    if (speed_y > 100)
                    {
                        speed_y = 100;
                    }
                }
                else if (std::abs(target_x) > 0.25)
                {
                    speed_x = -(1/0.5) * target_x * 100;
                    if (speed_x > 100)
                        speed_x = 100;
                }
                // if (std::abs(target_y) > 0.15)
                // {
                //     speed_y = -target_y * 100 * 2;
                //     if (speed_y > 100)
                //         speed_y = 100;
                //     print(YELLOW, "speed_y: " << speed_y);
                // }
                speed_y = std::abs(speed_x) > std::abs(speed_y) ? 0 : speed_y;
                speed_x = std::abs(speed_x) > std::abs(speed_y) ? speed_x : 0;
                while(!land)
                {
                    print(YELLOW, "speed_x:" << speed_x << "speed_y: " << speed_y);
                    print(RED, "target_loc_0: " << target_loc.at(0));
                    server.sendMsg(1, 0, speed_y, speed_x, 0, -100);
                    server.recvMsg();
                    land = server.is_landed(land);
                }
                print(BLUE, "land!!!!!!!!!!!!!");
                // 如果目标进入抓取阈值框, 转case13 (坐底)进行抓取; 否则记录目标相对与图像中心的方向
                target_x = float(target_loc.at(0)) / vis_size.width - 0.5;
                // print(RED, target_loc.at(0));
                target_y = float(target_loc.at(1)) / vis_size.height - 0.8;
                // target_x = std::abs(float(target_loc.at(0)) / vis_size.width - 0.5);
                // target_y = std::abs(float(target_loc.at(1)) / vis_size.height - 0.8);
                speed_x = 0;
                speed_y = 0;
                if(std::abs(target_x) > 0.25 || std::abs(target_y) > 0.2)
                {
                    print(RED, "No");
                    speed_x = -(1/0.5) * target_x * 100;
                    speed_y = -target_y * 100;
                    speed_x = std::abs(speed_x) > 40 ? speed_x : 40;
                    speed_x = std::abs(speed_x) > 100 ? 100 : speed_x;
                    speed_y = std::abs(speed_y) > 40 ? speed_y : 40;
                    speed_y = std::abs(speed_y) > 100 ? 100 : speed_y;
                    // if (speed_x > 100)
                    //     speed_x = 100;
                    // if (speed_y > 100)
                    // {
                    //     speed_y = 100;
                    // }
                    // speed_x = target_x / std::min(target_x, target_y) *  100;
                    // speed_y = target_y / std::min(target_x, target_y) *  100;
                }
                else
                {
                    print(RED, "land and grap");
                    rov_key = 13;
                    break;
                }
                // FIXME 向记录方向水平全速, 全速上浮x秒
                print(GREEN, "speed_x:" << speed_x << "speed_y: " << speed_y);
                if (std::abs(speed_x) > std::abs(speed_y))
                {
                    server.sendMsg(0, 0, 0, speed_x, 0, 99);
                }
                else
                {
                    server.sendMsg(0, 0, speed_y, 0, 0, 99);
                }
                
                delay_ms(1500);
            }
            break;
        default:
            server.sendMsg(SEND_SLEEP);
            break;
        }
    }
    for (unsigned char i = 0; i < 10; i++)
        server.sendMsg(SEND_SLEEP);
    print(RED, "QUIT: run_rov quit");
}