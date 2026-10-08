#include "../src/NetListenerLifecycle.h"
#include <cstdio>
#include <functional>
#include <memory>
#include <new>
#include <system_error>
#include <vector>
static int failures=0;
#define CHECK(c) do { if (!(c)) { ++failures; std::printf("FAIL line %d\n",__LINE__); } } while(0)
// A container seam injects exactly the push_back allocation failure used by production.
struct ThrowingConnections {
 void push_back(const std::shared_ptr<int>&) { throw std::bad_alloc(); }
};
int main() {
 using namespace DC::Net::detail;
 std::atomic<std::size_t> active{0};
 ThrowingConnections fail;
 auto holder=std::make_shared<int>(1);
 bool caught=false;
 try { registerListenerConnection(fail,active,holder); } catch(const std::bad_alloc&) {caught=true;}
 CHECK(caught); CHECK(active.load()==0); // stop cannot wait for a nonexistent worker
 std::vector<std::shared_ptr<int>> connections;
 registerListenerConnection(connections,active,holder);
 CHECK(connections.size()==1); CHECK(active.load()==1);
 connections.clear(); active.fetch_sub(1); CHECK(active.load()==0);

 std::atomic<bool> started{false},stopped{true};
 std::function<void()> handler=[]{};
 std::thread accept;
 caught=false;
 try {
  launchListenerAccept(started,stopped,handler,accept,[]()->std::thread {
   throw std::system_error(std::make_error_code(std::errc::resource_unavailable_try_again));
  });
 } catch(const std::system_error&) {caught=true;}
 CHECK(caught); CHECK(!started.load()); CHECK(stopped.load()); CHECK(!handler); CHECK(!accept.joinable());
 // Retrying after launch failure is valid and owns a real thread before publishing started.
 handler=[]{};
 std::atomic<bool> ran{false};
 launchListenerAccept(started,stopped,handler,accept,[&]{return std::thread([&]{ran=true;});});
 CHECK(started.load()); CHECK(!stopped.load()); CHECK(accept.joinable());
 accept.join(); CHECK(ran.load());
 std::printf("ListenerFailureRegressionTest: %d failures\n",failures);
 return failures?1:0;
}
