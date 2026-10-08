// CORE-1/2: deterministic lifecycle checks; prohibited destruction runs in
// isolated children, with a bounded parent wait (no detached production workers).
#include "ExecutionEngine.h"
#include "InferGraph.h"
#include "Tensor.hpp"
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <future>
#include <filesystem>
#include <fstream>
#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#ifdef _MSC_VER
#include <crtdbg.h>
#endif
#else
#include <sys/wait.h>
#include <unistd.h>
#include <signal.h>
#endif
using namespace DC;
using namespace std::chrono_literals;
static void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
static void waitReady(std::future<void>& f) {
    if (f.wait_for(5s) != std::future_status::ready) {
        std::cerr << "lifecycle regression timed out\n";
        std::_Exit(1); // don't block unwinding a deliberately deadlocked regression
    }
    f.get();
}
static void workerShutdown() {
    ThreadPool pool({1});
    std::promise<void> done;
    auto result = done.get_future();
    require(pool.submit([&] {
        try {
            require(pool.isWorkerThread(), "worker identity absent");
            bool rejected = false;
            try { pool.shutdown(); } catch (const std::logic_error&) { rejected = true; }
            require(rejected, "worker shutdown not rejected");
            require(pool.submit([] {}), "worker rejection changed running state");
            done.set_value();
        } catch (...) { done.set_exception(std::current_exception()); }
    }), "initial submit");
    waitReady(result);
    pool.shutdown();
    pool.shutdown();
}
static void payloadIdentity() {
    ThreadPool pool({1});
    std::promise<void> done;
    auto result = done.get_future();
    auto payload = std::shared_ptr<int>(new int(0), [&](int* p) {
        delete p;
        try {
            require(pool.isWorkerThread(), "identity ended before payload destruction");
            bool rejected = false;
            try { pool.shutdown(); } catch (const std::logic_error&) { rejected = true; }
            require(rejected, "payload destructor shutdown not rejected");
            done.set_value();
        } catch (...) { done.set_exception(std::current_exception()); }
    });
    require(pool.submit([payload = std::move(payload)] {}), "payload submit");
    waitReady(result);
    pool.shutdown();
}
static void concurrentPoolShutdown() {
    ThreadPool pool({1});
    std::promise<void> entered, go, done;
    auto enter = entered.get_future();
    auto gate = go.get_future().share();
    auto result = done.get_future();
    pool.submit([&] {
        entered.set_value();
        gate.wait();
        try {
            bool rejected = false;
            try { pool.shutdown(); } catch (const std::logic_error&) { rejected = true; }
            require(rejected, "concurrent worker shutdown not rejected");
            done.set_value();
        } catch (...) { done.set_exception(std::current_exception()); }
    });
    waitReady(enter);
    std::thread external([&] { pool.shutdown(); });
    const auto deadline = std::chrono::steady_clock::now() + 5s;
    while (pool.submit([] {}) && std::chrono::steady_clock::now() < deadline)
        std::this_thread::yield();
    if (std::chrono::steady_clock::now() >= deadline) std::_Exit(1);
    go.set_value(); // external holds drain lock and is joining this worker
    waitReady(result);
    external.join();
}
static void schedulerWorkers() {
    ResourceScheduler::resetInstance();
    auto scheduler = ResourceScheduler::instance();
    for (auto cls : {ResourceClass::Compute, ResourceClass::Operator, ResourceClass::System}) {
        std::promise<void> done;
        auto result = done.get_future();
        require(scheduler->submit(cls, [&] {
            try {
                bool shutdownRejected = false, resetRejected = false;
                try { scheduler->shutdown(); } catch (const std::logic_error&) { shutdownRejected = true; }
                try { ResourceScheduler::resetInstance(); } catch (const std::logic_error&) { resetRejected = true; }
                require(shutdownRejected && resetRejected, "scheduler worker rejection absent");
                require(!scheduler->isStopped(), "rejection stopped scheduler");
                require(ResourceScheduler::instance() == scheduler, "reset moved singleton before rejection");
                done.set_value();
            } catch (...) { done.set_exception(std::current_exception()); }
        }), "scheduler submit");
        waitReady(result);
    }
    ResourceScheduler::resetInstance();
}
static void concurrentSchedulerShutdown() {
    ResourceScheduler scheduler({1, 1, 1});
    std::promise<void> entered, go, done;
    auto enter = entered.get_future();
    auto gate = go.get_future().share();
    auto result = done.get_future();
    scheduler.submit(ResourceClass::System, [&] {
        entered.set_value(); gate.wait();
        try {
            bool rejected = false;
            try { scheduler.shutdown(); } catch (const std::logic_error&) { rejected = true; }
            require(rejected, "scheduler worker blocked behind outer shutdown lock");
            done.set_value();
        } catch (...) { done.set_exception(std::current_exception()); }
    });
    waitReady(enter);
    std::thread external([&] { scheduler.shutdown(); });
    const auto deadline = std::chrono::steady_clock::now() + 5s;
    while (!scheduler.isStopped() && std::chrono::steady_clock::now() < deadline)
        std::this_thread::yield();
    if (!scheduler.isStopped()) std::_Exit(1);
    go.set_value();
    waitReady(result);
    external.join();
}
static Node::Schema schema() {
    Node::Schema s;
    s.inputs = {Node::Port::in<float>("x")};
    s.outputs = {Node::Port::out<float>("y")};
    return s;
}
static void childCase(const std::string& mode) {
#ifdef _WIN32
    SetErrorMode(SEM_NOGPFAULTERRORBOX | SEM_FAILCRITICALERRORS);
#ifdef _MSC_VER
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
    _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ERROR, _CRTDBG_FILE_STDERR);
    _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
#endif
#endif
    std::set_terminate([] { std::_Exit(86); });
    if (mode == "pool") {
        std::atomic<bool> submitted{false};
        auto pool = std::make_unique<ThreadPool>(PoolConfig{1});
        require(pool->submit([&] {
            while (!submitted.load(std::memory_order_acquire)) std::this_thread::yield();
            std::set_terminate([] { std::_Exit(86); });
            pool.reset();
        }), "child pool submit");
        submitted.store(true, std::memory_order_release);
        std::this_thread::sleep_for(5s);
    } else if (mode == "scheduler") {
        std::atomic<bool> submitted{false};
        auto s = std::make_unique<ResourceScheduler>(SchedulerConfig{1, 1, 1});
        require(s->submit(ResourceClass::System, [&] {
            while (!submitted.load(std::memory_order_acquire)) std::this_thread::yield();
            std::set_terminate([] { std::_Exit(86); });
            s.reset();
        }), "child scheduler submit");
        submitted.store(true, std::memory_order_release);
        std::this_thread::sleep_for(5s);
    } else if (mode == "write" || mode == "nested") {
        auto s = std::make_shared<ResourceScheduler>();
        auto outer = std::make_unique<ExecutionEngine>(s);
        ExecutionEngine inner(s);
        outer->tryWriteTaskState("t", [&] {
            if (mode == "nested") inner.tryWriteTaskState("t", [&] { outer.reset(); });
            else outer.reset();
        });
    } else {
        auto s = std::make_shared<ResourceScheduler>();
        auto graph = std::make_unique<InferGraph>(s);
        graph->addNode(std::make_unique<Node>("test", "n", schema(), [&](Node::RunContext& ctx) -> Node::Result {
            std::set_terminate([] { std::_Exit(86); });
            if (mode == "node") graph.reset();
            if (mode == "rundone") return ctx.failure(Node::Status::ExecutionFailed, "failure triggers RunDone");
            auto t = Tensor::Create<float>();
            t = 1.0f;
            ctx.output("y", Value(std::make_unique<Tensor>(std::move(t))));
            return ctx.success();
        }));
        graph->bindInput("x", "n", "x");
        graph->bindOutput("y", "n", "y");
        if (mode != "node") graph->setTaskCompleteCallback([&](const std::string&) {
            std::set_terminate([] { std::_Exit(86); });
            graph.reset();
        });
        if (mode != "cancel") {
            auto t = Tensor::Create<float>(); t = 1.0f;
            graph->feedInput("t", "n", "x", Value(std::make_unique<Tensor>(std::move(t))));
        }
        graph->submitBound("t");
        if (mode == "cancel") graph->cancel("t"); // callback on nonworker, pendingRuns==0
        else std::this_thread::sleep_for(5s);
    }
    std::_Exit(2); // required diagnostic termination did not occur
}
static bool deathCase(const char* executable, const char* mode) {
    static unsigned long sequence = 0;
#ifdef _WIN32
    const auto parentPid = GetCurrentProcessId();
#else
    const auto parentPid = getpid();
#endif
    const auto started = std::chrono::steady_clock::now();
    auto log = std::filesystem::temp_directory_path() /
        ("dcinfer-core-" + std::to_string(parentPid) + "-" + std::to_string(++sequence) + "-" + mode + "-" +
         std::to_string(started.time_since_epoch().count()) + ".log");
    struct LogGuard {
        std::filesystem::path path;
        ~LogGuard() { std::error_code ec; std::filesystem::remove(path, ec); }
    } cleanup{log};
    auto reportFailure = [&](unsigned long exitCode, const char* outcome) {
        std::ifstream input(log);
        const std::string text((std::istreambuf_iterator<char>(input)), {});
        std::cerr << "death case " << mode << ": " << outcome << ", exit=" << exitCode
                  << ", elapsed_ms=" << std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::steady_clock::now() - started).count()
                  << ", expected lifecycle diagnostic, logfile=" << log.string()
                  << ", diagnostic=[" << text << "]\n";
    };
    auto diagnosed = [&] {
        std::ifstream input(log);
        const std::string text((std::istreambuf_iterator<char>(input)), {});
        return text.find("prohibited") != std::string::npos &&
            text.find(mode == std::string("pool") ? "ThreadPool:" :
                      mode == std::string("scheduler") ? "ResourceScheduler:" : "ExecutionEngine:") != std::string::npos;
    };
#ifdef _WIN32
    std::string command = std::string("\"") + executable + "\" --child " + mode + " \"" + log.string() + "\"";
    STARTUPINFOA startup{}; startup.cb = sizeof(startup);
    PROCESS_INFORMATION process{};
    if (!CreateProcessA(nullptr, command.data(), nullptr, nullptr, FALSE, 0, nullptr, nullptr, &startup, &process)) {
        reportFailure(GetLastError(), "CreateProcess failed"); return false;
    }
    const auto wait = WaitForSingleObject(process.hProcess, 10000);
    DWORD code = 0;
    if (wait != WAIT_OBJECT_0) {
        TerminateProcess(process.hProcess, 3);
        WaitForSingleObject(process.hProcess, 5000);
    }
    GetExitCodeProcess(process.hProcess, &code);
    CloseHandle(process.hThread); CloseHandle(process.hProcess);
    // MSVC's library CRT can terminate via abort (Debug exit 3) or fail-fast
    // rather than the executable's handler (independently linked runtimes).
    // Still require the specific lifecycle diagnostic and a completed bounded wait;
    // watchdog termination also uses 3 but cannot satisfy WAIT_OBJECT_0.
    const bool passed = wait == WAIT_OBJECT_0 && (code == 86 || code == 3 || code == 0xC0000409u) && diagnosed();
    if (!passed) reportFailure(code, wait == WAIT_OBJECT_0 ? "unexpected exit/diagnostic" : "child wait failed/timed out");
    return passed;
#else
    const auto pid = fork();
    if (pid == 0) { execl(executable, executable, "--child", mode, log.string().c_str(), nullptr); std::_Exit(4); }
    if (pid < 0) return false;
    const auto deadline = std::chrono::steady_clock::now() + 10s;
    int status = 0;
    while (std::chrono::steady_clock::now() < deadline) {
        if (waitpid(pid, &status, WNOHANG) == pid) return WIFEXITED(status) && WEXITSTATUS(status) == 86 && diagnosed();
        std::this_thread::sleep_for(5ms);
    }
    kill(pid, SIGKILL); waitpid(pid, &status, 0); return false;
#endif
}
int main(int argc, char** argv) {
    if (argc == 4 && std::string(argv[1]) == "--child") {
        if (!std::freopen(argv[3], "w", stderr)) return 4;
        childCase(argv[2]);
    }
    try {
        workerShutdown(); payloadIdentity(); concurrentPoolShutdown(); schedulerWorkers(); concurrentSchedulerShutdown();
        for (auto mode : {"pool", "scheduler", "node", "callback", "rundone", "cancel", "write", "nested"})
            require(deathCase(argv[0], mode), mode);
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n'; return 1;
    }
    std::cout << "CORE lifecycle regressions passed\n";
}
