#pragma once

// 极简 HTTP/1.1 Mock 远端，基于 POCO：DCNet 传输测试与 DCEngines 适配器测试共用。
// 单线程顺序处理；POST 请求体经 handler(path, body) 得到响应体。

#include <Poco/Net/ServerSocket.h>
#include <Poco/Net/SocketAddress.h>
#include <Poco/Net/StreamSocket.h>
#include <Poco/Timespan.h>

#include <atomic>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <string>
#include <thread>

class MockHttpServer {
public:
	/// 返回响应体；经 status 输出 HTTP 状态码。
	using Handler = std::function<std::string(const std::string& path, const std::string& body, int& status)>;

	MockHttpServer() = default;
	~MockHttpServer() { stop(); }

	/// 绑定 127.0.0.1 随机端口并启动监听线程；失败返回 -1。
	int start(Handler handler) {
		try {
			_socket.bind(Poco::Net::SocketAddress("127.0.0.1", 0), false);
			_socket.listen();
			_port = static_cast<int>(_socket.address().port());
		} catch (const std::exception&) {
			return -1;
		}
		_stop = false;
		_thread = std::thread([this, handler = std::move(handler)]() mutable { run(std::move(handler)); });
		return _port;
	}

	void stop() {
		_stop = true;
		if (_thread.joinable()) {
			try {
				_socket.close(); // 中断 accept 轮询
			} catch (...) {
			}
			_thread.join();
		}
	}

	int port() const { return _port; }

	/// 最近一次请求的原始头部，以 \r\n 分隔；响应返回后读取。
	std::string lastRequestHeaders() const {
		std::lock_guard lk(_hdrMutex);
		return _lastHeaders;
	}

private:
	void run(Handler handler) {
		for (;;) {
			if (_stop)
				break;
			// 轮询而非阻塞 accept：POSIX close 不唤醒阻塞 accept
			if (!_socket.poll(Poco::Timespan(0, 50 * 1000), Poco::Net::Socket::SELECT_READ))
				continue;
			Poco::Net::StreamSocket c;
			try {
				c = _socket.acceptConnection();
			} catch (...) {
				break;
			}
			serve(c, handler);
		}
		try {
			_socket.close();
		} catch (...) {
		}
	}

	void serve(Poco::Net::StreamSocket& c, Handler& handler) {
		std::string req;
		char buf[4096];
		int n;
		while ((n = c.receiveBytes(buf, sizeof(buf))) > 0) {
			req.append(buf, static_cast<size_t>(n));
			if (req.find("\r\n\r\n") != std::string::npos)
				break;
		}

		std::string path;
		{
			const size_t eol = req.find("\r\n");
			if (eol != std::string::npos) {
				const std::string line = req.substr(0, eol);
				const size_t sp1 = line.find(' ');
				const size_t sp2 = (sp1 == std::string::npos) ? std::string::npos : line.find(' ', sp1 + 1);
				if (sp1 != std::string::npos && sp2 != std::string::npos)
					path = line.substr(sp1 + 1, sp2 - sp1 - 1);
			}
		}

		size_t bodyLen = 0;
		{
			const size_t pos = req.find("Content-Length:");
			if (pos != std::string::npos)
				bodyLen = static_cast<size_t>(std::atoll(req.c_str() + pos + 15));
		}
		const size_t headerEnd = req.find("\r\n\r\n");
		{
			std::lock_guard lk(_hdrMutex);
			_lastHeaders = (headerEnd == std::string::npos) ? std::string() : req.substr(0, headerEnd);
		}
		std::string body = (headerEnd == std::string::npos) ? std::string() : req.substr(headerEnd + 4);
		while (body.size() < bodyLen) {
			n = c.receiveBytes(buf, sizeof(buf));
			if (n <= 0)
				break;
			body.append(buf, static_cast<size_t>(n));
		}

		int status = 200;
		std::string respBody;
		try {
			respBody = handler(path, body, status);
		} catch (const std::exception& e) {
			// handler 异常不得逃逸出服务线程，否则进程终止
			status = 500;
			respBody = std::string(R"({"error":{"code":"server_error","message":")") + e.what() + "\"}";
		}
		const char* statusText = (status == 200) ? "OK" : (status == 404) ? "Not Found" : "Internal Server Error";
		const std::string resp =
			"HTTP/1.1 " + std::to_string(status) + " " + statusText + "\r\n"
			"Content-Type: application/json\r\n"
			"Content-Length: " + std::to_string(respBody.size()) + "\r\n"
			"Connection: close\r\n\r\n" + respBody;
		// 循环补发：单次 sendBytes 允许短写，不补齐会制造截断假象
		try {
			std::size_t sent = 0;
			while (sent < resp.size()) {
				const int n = c.sendBytes(resp.data() + sent, static_cast<int>(resp.size() - sent));
				if (n <= 0)
					break;
				sent += static_cast<std::size_t>(n);
			}
		} catch (...) {
		}
		c.close();
	}

	int _port = -1;
	std::atomic<bool> _stop{false};
	std::thread _thread;
	Poco::Net::ServerSocket _socket;

	mutable std::mutex _hdrMutex;
	std::string _lastHeaders;
};
