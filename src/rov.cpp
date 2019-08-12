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
    const unsigned int backLog = 8;  // number of connections allowed on the incoming queue

    // addrinfo hints, *res, *p;
    addrinfo hints, *p;    // we need 2 pointers, res to hold and p to iterate over
    memset(&hints, 0, sizeof(hints));

    // for more explanation, man socket
    hints.ai_family   = AF_UNSPEC;    // don't specify which IP version to use yet
    hints.ai_socktype = SOCK_STREAM;  // SOCK_STREAM refers to TCP, SOCK_DGRAM will be?
    hints.ai_flags    = AI_PASSIVE;


    // man getaddrinfo
    int gAddRes = getaddrinfo(NULL, portNum, &hints, &res);
    if (gAddRes != 0) {
        std::cerr << gai_strerror(gAddRes) << "\n";
    }

    // std::cout << "Detecting addresses" << std::endl;

    unsigned int numOfAddr = 0;
    char ipStr[INET6_ADDRSTRLEN];    // ipv6 length makes sure both ipv4/6 addresses can be stored in this variable


    // Now since getaddrinfo() has given us a list of addresses
    // we're going to iterate over them and ask user to choose one
    // address for program to bind to
    for (p = res; p != NULL; p = p->ai_next) {
        void *addr;
        std::string ipVer;

        // if address is ipv4 address
        if (p->ai_family == AF_INET) {
            ipVer             = "IPv4";
            sockaddr_in *ipv4 = reinterpret_cast<sockaddr_in *>(p->ai_addr);
            addr              = &(ipv4->sin_addr);
            ++numOfAddr;
        }

        // if address is ipv6 address
        else {
            ipVer              = "IPv6";
            sockaddr_in6 *ipv6 = reinterpret_cast<sockaddr_in6 *>(p->ai_addr);
            addr               = &(ipv6->sin6_addr);
            ++numOfAddr;
        }

        // convert IPv4 and IPv6 addresses from binary to text form
        inet_ntop(p->ai_family, addr, ipStr, sizeof(ipStr));
        // std::cout << "(" << numOfAddr << ") " << ipVer << " : " << ipStr << std::endl;
    }

    // if no addresses found :(
    if (!numOfAddr) {
        std::cerr << "Found no host address to use\n";
    }

    // choose IPv4 address
    unsigned int choice = 0;
    bool madeChoice     = false;
    choice = 1;
    madeChoice = true;

    p = res;

    sockFD = socket(p->ai_family, p->ai_socktype, p->ai_protocol);
    // only for situation that the port restarted right away while sockets are still active on its port
    int opt = 1;
    setsockopt(sockFD, SOL_SOCKET, SO_REUSEADDR, (const void *)&opt, sizeof(opt));
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
    if(!is_new)
    {
        newFD = accept(sockFD, (sockaddr *) &client_addr, &client_addr_size);
        is_new = 1;
    }
    if (newFD == -1)
    {
        std::cerr << "Error while Accepting on socket\n";
        return;
    }
    memset(receive, 0, sizeof(receive));
    auto bytes_recv = recv(newFD, receive, 27, 0);

    if(receive[4] == '\xaa'){ isOneLeak = 1; }
    if(receive[7] == '\xaa'){ isTwoLeak = 1; }
    depth = (int(receive[8]) * 256 + int(receive[9]));  // the unit is cm
    // 此处adjust_rate即README中修正参数k
    depth = depth / adjust_rate;
    // std::cout << isOneLeak << std::endl;
    // std::cout << isTwoLeak << std::endl;
    // std::cout << depth << std::endl;
}

void TCP_Server::sendMsg(int move)
    // send move commands
    // moves:
    //    LIGHTS_ON FORWARD BACKWARD LEFT RIGHT TURN_LEFT TURN_RIGHT UP DOWN HALF_FORWARD HALF_BACKWARD HALF_LEFT HALF_RIGHT HALF_TURN_LEFT HALF_TURN_RIGHT HALF_UP HALF_DOWN SLEEP ADJUST_FORWARD ADJUST_BACKWARD ADJUST_LEFT ADJUST_RIGHT ADJUST_TURN_LEFT ADJUST_TURN_RIGHT
    //    0         1       2        3    4     5         6          7  8    9            10            11        12         13             14              15      16        17    18             19              20          21           22               23
{
    std::string response;
    switch(move)
    {
        case 0:
            response.assign(SEND_LIGHTS_ON, 27);
            break;
        case 1:
            response.assign(MOVE_FORWARD, 27);
            break;
        case 2:
            response.assign(MOVE_BACKWARD, 27);
            break;
        case 3:
            response.assign(MOVE_LEFT, 27);
            break;
        case 4:
            response.assign(MOVE_RIGHT, 27);
            break;
        case 5:
            response.assign(MOVE_TURN_LEFT, 27);
            break;
        case 6:
            response.assign(MOVE_TURN_RIGHT, 27);
            break;
        case 7:
            response.assign(MOVE_UP, 27);
            break;
        case 8:
            response.assign(MOVE_DOWN, 27);
            break;
        case 9:
            response.assign(MOVE_HALF_FORWARD, 27);
            break;
        case 10:
            response.assign(MOVE_HALF_BACKWARD, 27);
            break;
        case 11:
            response.assign(MOVE_HALF_LEFT, 27);
            break;
        case 12:
            response.assign(MOVE_HALF_RIGHT, 27);
            break;
        case 13:
            response.assign(MOVE_HALF_TURN_LEFT, 27);
            break;
        case 14:
            response.assign(MOVE_HALF_TURN_RIGHT, 27);
            break;
        case 15:
            response.assign(MOVE_HALF_UP, 27);
            break;
        case 16:
            response.assign(MOVE_HALF_DOWN, 27);
            break;
        case 17:
            response.assign(SEND_SLEEP, 27);
            break;
        case 18:
            response.assign(MOVE_ADJUST_FORWARD, 27);
            break;
        case 19:
            response.assign(MOVE_ADJUST_BACKWARD, 27);
            break;
        case 20:
            response.assign(MOVE_ADJUST_LEFT, 27);
            break;
        case 21:
            response.assign(MOVE_ADJUST_RIGHT, 27);
            break;
        case 22:
            response.assign(MOVE_ADJUST_TURN_LEFT, 27);
            break;
        case 23:
            response.assign(MOVE_ADJUST_TURN_RIGHT, 27);
            break;
        default:
            response.assign(MOVE_SLEEP, 27);
            break;
    }
    // send call sends the data you specify as second param and it's length as 3rd param, also returns how many bytes were actually sent
    // auto bytes_sent = send(newFD, response.data(), response.length(), 0);
    // response.erase(std::remove_if(response.begin(), response.end(), ::isspace), response.end());  // remove spaces
    auto bytes_sent = send(newFD, response.data(), response.length(), 0);
}

bool TCP_Server::is_landed(bool land_flag)
    // 结合上一时刻是否位于海底和深度波动判断当前是否位于海底
{
    depth_diff = depth - pre_depth;
    pre_depth = depth;
    // 更新海底深度
    if (land_flag)
    {
        max_depth = depth;
        print(BOLDGREEN, "ROV: update max depth = " << max_depth);
    }
    // 深度持续稳定时间计时
    if (depth_diff < depth_diff_thresh){
        land_count++;
    }
    // 当深度变化幅度超过阈值时判定为不在海底并归零深度持续稳定时间
    else {
        land_count = 0;
        return 0;
    }
    // land_count超过阈值count_thresh时判定为坐底
    // 当land_count和count_thresh过小时会产生噪声
    if (land_count >= count_thresh){
        print(BOLDGREEN, "ROV: landed at " << depth);
        return 1;
    }

}

TCP_Server server;
extern bool run_rov_flag;
extern int rov_key, send_byte;
extern bool rov_half_speed, land, manual_stop, grasping_done;
extern std::vector<int> target_loc;
extern cv::Size vis_size;

void run_rov(){
    //    bool first_diving = true;
    // float cruising_altitude = 40.0;
    //    int floating_stable_count = 0;
    //    int aming_stable_count = 0;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 5);
    // 确认上位机与ROV通信建立成功并开灯
    if (run_rov_flag > 0) {
        print(BOLDGREEN, "ROV: try to first receive");
        server.recvMsg();
        print(BOLDGREEN, "ROV: first receive done, current depth: " << server.depth);
        server.sendMsg(SEND_LIGHTS_ON);
    }
    // 进入ROV动作控制循环, 整个比赛过程中都应当处于这个循环中
    while(run_rov_flag) {
        // 键盘键值与ROV动作映射
        switch (rov_key) {
            case 105: // k
                if (rov_half_speed) server.sendMsg(SEND_HALF_FORWARD);
                else server.sendMsg(SEND_FORWARD);
                break;
            case 106: // j
                if (rov_half_speed) server.sendMsg(SEND_HALF_TURN_LEFT);
                else server.sendMsg(SEND_TURN_LEFT);
                break;
            case 107: // k
                if (rov_half_speed) server.sendMsg(SEND_HALF_BACKWARD);
                else server.sendMsg(SEND_BACKWARD);
                break;
            case 108: // l
                if (rov_half_speed) server.sendMsg(SEND_HALF_TURN_RIGHT);
                else server.sendMsg(SEND_TURN_RIGHT);
                break;
            case 44: // ,
                server.sendMsg(SEND_DOWN);
                break;
            case 46: // .
                if (rov_half_speed) server.sendMsg(SEND_HALF_UP);
                else server.sendMsg(SEND_UP);
                break;
            case 59: // ;
                // 坐底. 从这一步开始为自主控制.
                print(BOLDGREEN, "ROV: diving !!!");
                grasping_done = false;
                // 当未人为操作且软体臂抓取未完成时持续坐底
                while(!manual_stop && !grasping_done)
                {
                    server.sendMsg(SEND_DOWN);
                    server.recvMsg();
                    // 用depth > 0滤掉因为上位机端刷新数据过快而并非每次都能接收到数据产生的0
                    // (每次刷新数据前都会用memset归零接收ROV数据的变量)
                    if (server.depth > 0)
                        // 判定是否到达海底
                    {
                        // depth_diff = server.depth - pre_depth;
                        // pre_depth = server.depth;
                        // 更新海底深度
                        // if (land) {
                        //                            print(BOLDGREEN, "ROV: update max depth = " << max_depth);
                        // max_depth = server.depth;
                        // }
                        // 深度持续稳定时间计时
                        // if (depth_diff < depth_diff_thresh){
                        // land_count++;
                        // }
                        // 当深度变化幅度超过阈值时判定为不在海底并归零深度持续稳定时间
                        // else {
                        // land = false;
                        // land_count = 0;
                        // }
                        // land_count超过阈值count_thresh时判定为坐底
                        // 当land_count和count_thresh过小时会产生噪声
                        // if (land_count >= count_thresh){
                        // print(BOLDGREEN, "ROV: landed at " << server.depth);
                        // land = true;
                        // }
                        land = is_landed(land);
                    }
                }
                land_count = 0;
                land = false;  // 结束坐底
                // first_diving = false;
                if(manual_stop){ rov_key = 99; }
                // 开始上浮并定深
                else { rov_key = 39; }
                break;
            case 39: // '
                // 定深
                print(BOLDBLUE,  "ROV: try to stably floating");
                // 全速上浮3个单位时间, 悬停2个单位时间等ROV静止后获取当前深度
                for (unsigned char i=0; i<100; i++)
                    server.sendMsg(SEND_UP);
                delay(3);
                for (unsigned char i=0; i<100; i++)
                    server.sendMsg(SEND_SLEEP);
                delay(2);
                while(true) {
                    server.recvMsg();
                    if (server.depth > 0){
                        print(BOLDBLUE, "ROV: floating at " << server.depth);
                        break;
                    }
                }
                // init_state();
                if(manual_stop) rov_key = 99;
                else rov_key = 47;
                break;
            case 47:  // /
                // 视野内无目标时遍历水域
                // TODO
                print(BOLDMAGENTA, "ROV: cruising");
                while((!manual_stop))
                {
                    // 视野内有扇贝或海参时跳转case43开始微调并坐底
                    if (target_loc.at(2) != 0 && target_loc.at(3) != 0)
                    {
                        rov_key = 43;
                        break;
                    }
                    // delay(5);
                    // switch (dis(gen))
                    // {
                    // case 0: print(BOLDMAGENTA, "ROV: random turn right"); server.sendMsg(SEND_HALF_TURN_RIGHT); break;
                    // case 1: print(BOLDMAGENTA, "ROV: random turn right"); server.sendMsg(SEND_HALF_TURN_LEFT); break;
                    // case 2: print(BOLDMAGENTA, "ROV: random forward"); server.sendMsg(SEND_HALF_FORWARD); break;
                    // case 3: print(BOLDMAGENTA, "ROV: random backward"); server.sendMsg(SEND_HALF_BACKWARD); break;
                    // case 4: print(BOLDMAGENTA, "ROV: random left"); server.sendMsg(SEND_HALF_LEFT); break;
                    // case 5: print(BOLDMAGENTA, "ROV: random right"); server.sendMsg(SEND_HALF_RIGHT); break;
                    // }
                }
            case 43:  // +
                // 实时微调水平位置并全速下潜
                while((!manual_stop))
                {
                    // TODO
                    delay(1);
                    // if (target_loc.at(0) < (float) vis_size.width * 0.2) {
                    // print(BOLDMAGENTA, "ROV: left");
                    // server.sendMsg(SEND_HALF_LEFT);
                    // } else if (target_loc.at(0) > (float) vis_size.width * 0.2) {
                    // print(BOLDMAGENTA, "ROV: right");
                    // server.sendMsg(SEND_HALF_RIGHT);
                    // } else if (target_loc.at(1) < (float) vis_size.height * 0.3) {
                    // print(BOLDMAGENTA, "ROV: forward");
                    // server.sendMsg(SEND_HALF_FORWARD);
                    // } else if (target_loc.at(1) > (float) vis_size.height * 0.3) {
                    // print(BOLDMAGENTA, "ROV: backward");
                    // server.sendMsg(SEND_HALF_BACKWARD);
                    // } else {
                    // print(BOLDMAGENTA, "ROV: down");
                    // break;
                    // }
                    // 当目标丢失时跳回定深
                    // FIXME: 直接跳转导致定深高度可能高了一些.
                    // 当目标在阈值内时全速下潜
                    // 当目标在视野内但漂移出阈值框, 全速坐底的同时微调水平位置
                    // 当判定为坐底时break并跳到case59 定深
                    // FIXME: 这样的话当坐底后目标在阈值框外也会尝试抓取, 但因为阈值框较小因此确实应当尝试抓取
                }
                if(manual_stop) rov_key = 99;
                else rov_key = 59;
                break;
            case 99: // c
                // 急停
                server.sendMsg(SEND_SLEEP);
                break;
            default:
                server.sendMsg(SEMD_SLEEP);
                break;
        }
    }
    server.sendMsg(SEND_SLEEP);
    print(WHITE, "ROV: run_rov quit");
}
