#include <thread>
// for socket
#include <netinet/in.h>
#include <sys/socket.h>
// local
#include "detector.h"
// detector.h必须在color.h之前, 不然和torch里的print函数冲突
#include "color.h"
#include "visual_server.h"

extern bool threads_quit_flag;
extern detector::Visual_info visual_info;

void server::recvMsg(int newFD)
{
    char receive[10];
    memset(receive, 0, sizeof(receive));
    recv(newFD, receive, 10, 0);
    std::string arm_is_working{receive};
    if (arm_is_working == "false")
    {
        visual_info.arm_is_working = false;
    }
    else if (arm_is_working == "true")
    {
        visual_info.arm_is_working = true;
    }
}

void server::communicate(int newFD)
{
    server::recvMsg(newFD);
    // send YAML
    std::stringstream ss;
    ss << "target:" << std::endl;
    ss << "  has_target: " << std::boolalpha << visual_info.has_target << std::endl;
    ss << "  class: " << visual_info.target_class << std::endl;
    ss << "  id: " << visual_info.target_id << std::endl;
    ss << "  center:" << std::endl;
    ss << "    x: " << visual_info.target_center.x << std::endl;
    ss << "    y: " << visual_info.target_center.y << std::endl;
    ss << "  shape:" << std::endl;
    ss << "    width: " << visual_info.target_shape.x << std::endl;
    ss << "    height: " << visual_info.target_shape.y << std::endl;
    ss << "arm:" << std::endl;
    ss << "  arm_is_working: " << std::boolalpha << visual_info.arm_is_working << std::endl;
    ss << "  has_marker: " << std::boolalpha << visual_info.has_marker << std::endl;
    ss << "  position:" << std::endl;
    ss << "    x: " << visual_info.marker_position.x << std::endl;
    ss << "    y: " << visual_info.marker_position.y << std::endl;
    std::string msg = ss.str();
    send(newFD, msg.data(), msg.length(), 0);
    close(newFD);
}

void server::server_start()
{
    // 创建socket文件描述符
    int sockFD, newFD;
    sockFD = socket(AF_INET, SOCK_STREAM, IPPROTO_IP);
    struct sockaddr_in server_addr, client_addr;
    socklen_t client_addr_size = sizeof(client_addr);
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET; // 使用IPv4地址
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(LOCAL_PORT);
    // only for situation that the port restarted right away while sockets are still active on its port
    int opt = 1;
    setsockopt(sockFD, SOL_SOCKET, SO_REUSEADDR, (const void *)&opt, sizeof(opt));
    // 绑定端口
    int bindR = bind(sockFD, (struct sockaddr *)&server_addr, sizeof(server_addr));
    if (bindR == -1)
    {
        print(BOLDRED, "[ERROR] Cannot bind socket");
        close(sockFD);
        print(BOLDBLUE, "[Server] quit");
        return;
    }
    // finally start listening for connections on our socket
    int listenR = listen(sockFD, BACKLOG);
    print(BOLDBLUE, "[Server] start");
    while (!threads_quit_flag)
    {
        newFD = accept(sockFD, (struct sockaddr *)&client_addr, &client_addr_size);
        if (newFD == -1)
        {
            print(BOLDRED, "[ERROR] Cannot create client socket");
            break;
        }
        std::thread th(server::communicate, newFD);
        th.detach();
    }
    close(sockFD);
    print(BOLDBLUE, "[Server] quit");
}
