#pragma once
// Private exception-safety seams shared by production listener and fault-injection tests.
// Callers hold the relevant lifecycle/connection mutex; no public test hook or global state.
#include <atomic>
#include <thread>
#include <utility>
namespace DC::Net::detail {
template<class Connections, class Holder>
void registerListenerConnection(Connections& connections, std::atomic<std::size_t>& active, const Holder& holder) {
 connections.push_back(holder); // potentially throwing allocation precedes irreversible accounting
 active.fetch_add(1, std::memory_order_acq_rel);
}
template<class Handler, class Launch>
void launchListenerAccept(std::atomic<bool>& started, std::atomic<bool>& stopped,
                          Handler& handler, std::thread& thread, Launch&& launch) {
 stopped.store(false);
 try {
  thread = std::forward<Launch>(launch)();
  started.store(true); // publish only after thread ownership is established
 } catch (...) {
  started.store(false);
  stopped.store(true);
  handler = {};
  throw;
 }
}
}
