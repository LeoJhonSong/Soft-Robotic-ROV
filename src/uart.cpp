//
// Created by sean on 7/21/19.
//

#include <iostream>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <termios.h>
#include <uart.h>

Uart::Uart(std::string port, int baud_rate){
    port_status = NOT_INIT;
    file_descriptor = 0;
    this->port = port;
    this->baud_rate = baud_rate;
    flow_ctl = 0;
    databits = 8;
    stopbits = 1;
    parity = 'N';
    if(openFile() == false){
        return;
    }
    if(initPort() == false){
        return;
    }
    else{
        port_status = INITED;
    }
}

Uart::~Uart(){
    closeFile();
}

bool Uart::openFile(void){
    int fd;

    if( file_descriptor > 0 ){
        return true;
    }

    std::string dev_filename = "/dev/";
    dev_filename = dev_filename + port;
    // 以读写模式, 不将该tty设备作为控制终端, 无延时模式
    fd = open(dev_filename.c_str(), O_RDWR|O_NOCTTY|O_NDELAY);
    for(int i = 0; fd < 0 && i < 1000; i++){
        fd = open(dev_filename.c_str(), O_RDWR|O_NOCTTY|O_NDELAY);
    }
    if(-1 == fd){
        std::cout << "Can't Open Serial Port" << std::endl;
        return false;
    }

    if(fcntl(fd, F_SETFL, 0) < 0){
        std::cout << "fcntl failed!" << std::endl;
        return false;
    }
    else{
        std::cout << "fcntl = " << fcntl(fd, F_SETFL, 0) << std::endl;
    }

    if(0 == isatty(STDIN_FILENO)){
        std::cout << "standard input is not a terminal device" << std::endl;
        return false;
    }
    else{
        std::cout << "isatty success!" << std::endl;
    }

    file_descriptor = fd;
    std::cout << "fd->open=" << file_descriptor << std::endl;
    return true;
}

bool Uart::closeFile(void){
    close(file_descriptor);
    return true;
}

bool Uart::initPort(void){
    unsigned int i;

    int speed_arr[] = {B115200, B57600, B38400, B19200,\
                       B9600,   B4800,  B2400,  B1200};
    int name_arr[]  = {115200,  57600,  38400,  19200,\
                       9600,    4800,   2400,   1200};

    struct termios options;

    if( tcgetattr(file_descriptor, &options) != 0){
        std::cout << "Setup Serial Failed" << std::endl;
        return false;
    }

    for(i = 0; i < sizeof(speed_arr)/sizeof(int); i++){
        if( baud_rate == name_arr[i] ){
            cfsetispeed(&options, speed_arr[i]);
            cfsetospeed(&options, speed_arr[i]);
            break;
        }
    }
    if(i == sizeof(speed_arr)/sizeof(int)){
        std::cout << "Unsupported Baud Rate" << std::endl;
        return false;
    }

    options.c_cflag |= CLOCAL;
    options.c_cflag |= CREAD;

    switch(flow_ctl){
        case 0:
            options.c_cflag &= ~CRTSCTS;
            break;
        case 1:
            options.c_cflag |= CRTSCTS;
            break;
        case 2:
            options.c_cflag |= IXON | IXOFF | IXANY;
            break;
        default:
            std::cout << "Unsupported Flow Control Parmeters" << std::endl;
            return false;
    }

    options.c_cflag &= ~CSIZE;

    switch(databits){
        case 5:
            options.c_cflag |= CS5;
            break;
        case 6:
            options.c_cflag |= CS6;
            break;
        case 7:
            options.c_cflag |= CS7;
            break;
        case 8:
            options.c_cflag |= CS8;
            break;
        default:
            std::cout << "Unsupported Data Bits Type" << std::endl;
            return false;
    }

    switch(parity){
        case 'n':
        case 'N':
            options.c_cflag &= ~PARENB;
            options.c_iflag &= ~INPCK;
            break;
        case 'o':
        case 'O':
            options.c_cflag |= (PARODD | PARENB);
            options.c_iflag |= INPCK;
            break;
        case 'e':
        case 'E':
            options.c_cflag |= PARENB;
            options.c_cflag &= ~PARODD;
            options.c_iflag |= INPCK;
            break;
        case 's':
        case 'S':
            options.c_cflag &= ~PARENB;
            options.c_cflag &= ~CSTOPB;
            break;
        default:
            std::cout << "Unsupported Parity Mode" << std::endl;
            return false;
    }

    options.c_oflag &= ~OPOST;

    options.c_cc[VTIME] = 1;
    options.c_cc[VMIN] = 1;

    tcflush( file_descriptor, TCIFLUSH );

    if( tcsetattr( file_descriptor, TCSANOW, &options ) != 0 ){
        std::cout << "COM Setting Up Error!!!" << std::endl;
        return false;
    }

    return true;
}

int Uart::recv(char * rcv_buf, int data_len){
    if(port_status != INITED){
        return 0;
    }

    int len, fs_sel;
    fd_set fs_read;

    struct timeval time;

    FD_ZERO(&fs_read);
    FD_SET(file_descriptor, &fs_read);

    time.tv_sec = 10;
    time.tv_usec = 0;

    fs_sel = select(file_descriptor + 1, &fs_read, NULL, NULL, &time);

    if(fs_sel){
        len = read(file_descriptor, rcv_buf, data_len);
        return len;
    }
    else{
        return 0;
    }
}

int Uart::send(char * send_buf, int data_len){
    if(port_status != INITED){
        return 0;
    }

    int ret;

    ret = write(file_descriptor, send_buf, data_len);

    if(data_len == ret){
        return ret;
    }
    else{
        return 0;
    }
}

int Uart::send(std::vector<char> send_list){
    int ret = 0;
    char * send_buf = new char [send_list.size()];

    for(unsigned int i = 0; i < send_list.size(); i++){
        send_buf[i] = send_list[i];
    }

    ret = send(send_buf, send_list.size());

    delete [] send_buf;

    return ret;
}

void Uart::logging(std::string str){
    if(logging_level > 1)
    {
        std::cout << str << std::endl;
    }
}

void Uart::errorOut(std::string str){
    if(logging_level > 0)
    {
        std::cout << str << std::endl;
    }
}
