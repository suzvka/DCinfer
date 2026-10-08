#include "DCNet/NetListener.h"
#include "DCNet/NetTransport_Http.h"
#include "NodeException.h"
#include <Poco/Net/SocketAddress.h>
#include <Poco/Net/StreamSocket.h>
#include <Poco/Timespan.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <thread>
using namespace DC::Net;
static std::atomic<int> failures{0};
#define CHECK(c) do { if (!(c)) { ++failures; std::printf("FAIL line %d\n", __LINE__); } } while(0)
static void post(int port, const std::string& raw, int status) {
 Poco::Net::StreamSocket c;
 c.connect(Poco::Net::SocketAddress("127.0.0.1", static_cast<unsigned short>(port)), Poco::Timespan(2,0));
 c.setReceiveTimeout(Poco::Timespan(2,0));
 c.sendBytes(raw.data(), static_cast<int>(raw.size()));
 std::string response; char buf[512]; int n;
 while (response.find("\r\n\r\n") == std::string::npos && (n=c.receiveBytes(buf,sizeof(buf)))>0) response.append(buf,n);
 CHECK(response.rfind("HTTP/1.1 " + std::to_string(status) + " ",0)==0);
}
static void preBody() {
 auto l=makeHttpListener(); NetServerEndpoint ep; ep.authToken="Bearer secret"; ep.maxRequestBody=16; ep.maxBufferedBodyBytes=16;
 l->bind(ep); std::atomic<int> calls{0};
 l->start([&](const std::string&,const std::string&){ ++calls; return WireResponse{200,"{}"}; });
 post(l->port(),"POST /v1/infer HTTP/1.1\r\nContent-Length: 999999999\r\n\r\n",401);
 post(l->port(),"POST /v1/infer HTTP/1.1\r\nAuthorization: Bearer secret\r\nContent-Length: 17\r\n\r\n",413);
 post(l->port(),"POST /v1/infer HTTP/1.1\r\nTransfer-Encoding: chunked\r\nContent-Length: 0\r\n\r\n",400);
 post(l->port(),"POST /v1/infer HTTP/1.1\r\nExpect: 100-continue\r\nContent-Length: 0\r\n\r\n",400);
 CHECK(calls==0); l->stop();
 auto external=makeHttpListener(); ep.listenHost="0.0.0.0"; bool rejected=false;
 try { external->bind(ep); } catch (...) { rejected=true; } CHECK(rejected);
}
static void lifecycle() {
 auto l=makeHttpListener(); NetServerEndpoint ep; l->bind(ep);
 std::atomic<bool> entered{false},release{false},rejected{false};
 l->start([&](const std::string&,const std::string&){ entered=true; while(!release.load()) std::this_thread::yield();
 try {l->stop();} catch(const DC::NodeException&) {rejected=true;} return WireResponse{200,"{}"}; });
 int port=l->port(); std::thread client([&]{post(port,"POST /v1/infer HTTP/1.1\r\nContent-Length: 0\r\n\r\n",200);});
 while(!entered.load()) std::this_thread::yield(); std::atomic<int> drained{0};
 std::thread a([&]{l->stop();++drained;}); std::thread b([&]{l->stop();++drained;});
 std::this_thread::sleep_for(std::chrono::milliseconds(50)); CHECK(drained==0); release=true;
 client.join(); a.join(); b.join(); CHECK(rejected); CHECK(drained==2); l.reset();
}
static void genericError() {
 auto l=makeHttpListener(); NetServerEndpoint ep; std::string diagnostic;
 ep.diagnosticSink=[&](const std::string& event){diagnostic=event;}; l->bind(ep);
 l->start([](const std::string&,const std::string&)->WireResponse {throw std::runtime_error("C:/secret/model PRIVATE\r\ninjected");});
 HttpTransport t; NetEndpoint client; client.host="127.0.0.1"; client.port=l->port(); client.requestPath="/infer";
 CHECK(t.connect(client).ok()); auto err=t.send("{}"); CHECK(!err.ok());
 CHECK(err.localMessage.find("correlation=")!=std::string::npos); CHECK(err.localMessage.find("PRIVATE")==std::string::npos);
 CHECK(err.localMessage.find("C:/secret")==std::string::npos); l->stop();
 CHECK(diagnostic.rfind("dcnet-",0)==0); CHECK(diagnostic.find("PRIVATE")==std::string::npos);
}
static void bodyBudgetRecovery() {
 auto l=makeHttpListener(); NetServerEndpoint ep; ep.maxRequestBody=16; ep.maxBufferedBodyBytes=16; ep.requestTimeout=std::chrono::milliseconds(500);
 l->bind(ep); l->start([](const std::string&,const std::string&){return WireResponse{200,"{}"};});
 Poco::Net::StreamSocket pending; pending.connect(Poco::Net::SocketAddress("127.0.0.1",static_cast<unsigned short>(l->port())));
 const std::string head="POST /v1/infer HTTP/1.1\r\nContent-Length: 16\r\n\r\n";
 pending.sendBytes(head.data(),static_cast<int>(head.size()));
 std::this_thread::sleep_for(std::chrono::milliseconds(50));
 post(l->port(),head,429);
 pending.close(); std::this_thread::sleep_for(std::chrono::milliseconds(100));
 post(l->port(),"POST /v1/infer HTTP/1.1\r\nContent-Length: 16\r\n\r\n0123456789012345",200);
 l->stop();
}
static void sharedReadDeadline() {
 auto l=makeHttpListener(); NetServerEndpoint ep; ep.requestTimeout=std::chrono::milliseconds(1000);
 std::atomic<int> calls{0}; l->bind(ep); l->start([&](const std::string&,const std::string&){++calls;return WireResponse{200,"{}"};});
 Poco::Net::StreamSocket c; c.connect(Poco::Net::SocketAddress("127.0.0.1",static_cast<unsigned short>(l->port())));
 c.setReceiveTimeout(Poco::Timespan(2,0));
 const auto start=std::chrono::steady_clock::now();
 const std::string first="POST /v1/infer HTTP/1.1\r\nContent-Length: 1\r\n";
 c.sendBytes(first.data(),static_cast<int>(first.size()));
 std::this_thread::sleep_for(std::chrono::milliseconds(700));
 c.sendBytes("\r\n",2); // Finish headers but withhold body after most of the shared budget.
 std::string response; char buf[512]; int n;
 while(response.find("\r\n\r\n")==std::string::npos && (n=c.receiveBytes(buf,sizeof(buf)))>0)response.append(buf,n);
 // Socket read timeout is caught by serveConnection and closes silently;
 // a locally exhausted budget instead returns the existing malformed-body 400.
 CHECK(response.empty() || response.rfind("HTTP/1.1 400 ",0)==0);
 CHECK(std::chrono::steady_clock::now()-start<std::chrono::milliseconds(1450));
 CHECK(calls==0); c.close(); l->stop();
}
int main(){preBody();lifecycle();genericError();bodyBudgetRecovery();sharedReadDeadline(); std::printf("ListenerSecurityRegressionTest: %d failures\n",failures.load());return failures.load()?1:0;}
