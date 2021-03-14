#ifndef RESDET_SERVER_H
#define RESDET_SERVER_H

namespace server
{
#define LOCAL_PORT 9090 // local port to listen by server
#define BACKLOG 8       // number of connections allowed on the incoming queue

void recvMsg(int newFD);
void communicate(int newFD);
void server_start();
} // namespace server
#endif // !RESDET_SERVER_H