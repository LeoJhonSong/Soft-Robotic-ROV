//
// Created by SONGZhuHeng on 2019/7/30.
//

#include "rov.h"

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
    depth = depth / adjust_rate;
    // std::cout << isOneLeak << std::endl;
    // std::cout << isTwoLeak << std::endl;
    // std::cout << depth << std::endl;
}

void TCP_Server::sendMsg(int move)
// send move commands
// moves:
//    FORWARD BACKWARD LEFT RIGHT TURN_LEFT TURN_RIGHT UP DOWN HALF_FORWARD HALF_BACKWARD HALF_LEFT HALF_RIGHT HALF_TURN_LEFT HALF_TURN_RIGHT HALF_UP HALF_DOWN
//    0       1        2    3     4         5    6     7  8    9            10            11        12         13             14              15
{
    std::string response;
    switch(move)
    {
        case 0:
            response.assign(MOVE_FORWARD, 27);
            break;
        case 1:
            response.assign(MOVE_BACKWARD, 27);
            break;
        case 2:
            response.assign(MOVE_LEFT, 27);
            break;
        case 3:
            response.assign(MOVE_RIGHT, 27);
            break;
        case 4:
            response.assign(MOVE_TURN_LEFT, 27);
            break;
        case 5:
            response.assign(MOVE_TURN_RIGHT, 27);
            break;
        case 6:
            response.assign(MOVE_UP, 27);
            break;
        case 7:
            response.assign(MOVE_DOWN, 27);
            break;
        case 8:
            response.assign(MOVE_HALF_FORWARD, 27);
            break;
        case 9:
            response.assign(MOVE_HALF_BACKWARD, 27);
            break;
        case 10:
            response.assign(MOVE_HALF_LEFT, 27);
            break;
        case 11:
            response.assign(MOVE_HALF_RIGHT, 27);
            break;
        case 12:
            response.assign(MOVE_HALF_TURN_LEFT, 27);
            break;
        case 13:
            response.assign(MOVE_HALF_TURN_RIGHT, 27);
            break;
        case 14:
            response.assign(MOVE_HALF_UP, 27);
            break;
        case 15:
            response.assign(MOVE_HALF_DOWN, 27);
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

extern bool run_rov_flag;
extern int rov_key, send_byte;
extern bool rov_half_speed, land, dive_ready;
void run_rov(){
    TCP_Server server;
    int land_count = 0;
    int land_count_thresh = 30;
    float pre_depth = 0;
    float depth_diff_thresh = 3.0;
    float depth_diff = 0.0;
    if (run_rov_flag > 0) {
        std::cout << "rov_runner: try to first receive" << std::endl;
        server.recvMsg();
        std::cout << "rov_runner: first receive done, current depth: " << server.depth << std::endl;
    }
    while(run_rov_flag) {
        switch (rov_key) {
            case 105: // up
//                std::cout << "rov_runner: forward" << std::endl;
//                std::cout << "rov_runner: " << time(nullptr) << std::endl;
                if (rov_half_speed) server.sendMsg(SEND_HALF_FORWARD);
                else server.sendMsg(SEND_FORWARD);
//                sleep(1);
//                server.sendMsg(SEND_SLEEP);
                break;
            case 106: // left
//                std::cout << "rov_runner: left" << std::endl;
                if (rov_half_speed) server.sendMsg(SEND_HALF_TURN_LEFT);
                else server.sendMsg(SEND_TURN_LEFT);
//                sleep(1);
//                server.sendMsg(SEND_SLEEP);
                break;
            case 107: // down
//                std::cout << "rov_runner: backward" << std::endl;
                if (rov_half_speed) server.sendMsg(SEND_HALF_BACKWARD);
                else server.sendMsg(SEND_BACKWARD);
//                sleep(1);
//                server.sendMsg(SEND_SLEEP);
                break;
            case 108: // right
//                std::cout << "rov_runner: right" << std::endl;
                if (rov_half_speed) server.sendMsg(SEND_HALF_TURN_RIGHT);
                else server.sendMsg(SEND_TURN_RIGHT);
//                sleep(1);
//                server.sendMsg(SEND_SLEEP);
                break;
            case 44: // ,
//                std::cout << "rov_runner: down" << std::endl;
//                if (rov_half_speed) server.sendMsg(SEND_HALF_DOWN);
                server.sendMsg(SEND_DOWN);
//                sleep(1);
//                server.sendMsg(SEND_SLEEP);
                break;
            case 46: // .
//                std::cout << "rov_runner: up" << std::endl;
                if (rov_half_speed) server.sendMsg(SEND_HALF_UP);
                else server.sendMsg(SEND_UP);
//                sleep(1);
//                server.sendMsg(SEND_SLEEP);
                break;
            case 59: // ;
                std::cout << "rov_runner: move down until dive_ready=True" << std::endl;
                dive_ready = false;
                while((!dive_ready) && land_count<200) {
                    server.sendMsg(SEND_DOWN);
                    server.recvMsg();
                    if (server.depth > 0) {
//                        std::cout << "rov_runner: current depth: " << server.depth << std::endl;
                        depth_diff = server.depth - pre_depth;
                        pre_depth = server.depth;
                    }
                    if (depth_diff < depth_diff_thresh) land_count++;
                    else if(land) {
                        land = false;
                        land_count = 0;
                    }
                    if (land_count >= land_count_thresh) land = true;
                }
                dive_ready = true;
                land = false;
                send_byte = -1;
                land_count = 0;
                break;
            case 99: // c
                server.sendMsg(SEND_SLEEP);
                break;
        }
    }
    server.sendMsg(SEND_SLEEP);
    std::cout << "run_rov quit" << std::endl;
}
