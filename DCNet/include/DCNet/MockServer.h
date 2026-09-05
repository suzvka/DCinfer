#pragma once

// 极简 HTTP/1.1 测试服务（POCO）：DCNet 传输测试及 DCEngines 协议适配器
// 测试共用的 Mock 远端（经 DCNet::DCNet 传递包含）。跨平台（Poco::Net）。
// 单线程顺序处理；POST 请求体 → handler(path, body) → 响应体。

#include <Poco/Net/ServerSocket.h>
#include <Poco/Net/SocketAddress.h>
#include <Poco/Net/StreamSocket.h>
#include <Poco/Timespan.h>

#include <atomic>
#include <cstdlib>
#include <functional>
#include <string>
#include <thread>

class MockHttpServer {
public:
	/// path: 请求路径（如 /v1/infer）；返回响应体，经 status 输出 HTTP 状态码。
	using Handler = std::function<std::string(const std::string& path, const std::string& body, int& status)>;

	MockHttpServer() = default;
	~MockHttpServer() { stop(); }

	/// 绑定 127.0.0.1:0（随机端口）并启动监听线程；返回实际端口，失败返回 -1。
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

	/// 停止监听并等待处理线程退出。
	void stop() {
		_stop = true;
		if (_thread.joinable()) {
			try {
				_socket.close(); // 中断 accept（轮询循环在 50ms 内感知 _stop）
			} catch (...) {
			}
			_thread.join();
		}
	}

	int port() const { return _port; }

private:
	void run(Handler handler) {
		for (;;) {
			if (_stop)
				break;
			// 轮询而非阻塞 accept：stop 时 close+join 跨平台安全（POSIX close 不唤醒阻塞 accept）
			if (!_socket.poll(Poco::Timespan(0, 50 * 1000), Poco::Net::Socket::SELECT_READ))
				continue;
			Poco::Net::StreamSocket c;
			try {
				c = _socket.acceptConnection();
			} catch (...) {
				break; // socket 已关闭（stop）
			}
			serve(c, handler);
		}
		try {
			_socket.close();
		} catch (...) {
		}
	}

	void serve(Poco::Net::StreamSocket& c, Handler& handler) {
		// 读取请求（头部 + body）
		std::string req;
		char buf[4096];
		int n;
		while ((n = c.receiveBytes(buf, sizeof(buf))) > 0) {
			req.append(buf, static_cast<size_t>(n));
			if (req.find("\r\n\r\n") != std::string::npos)
				break;
		}

		// 请求行 → path
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

		// Content-Length → body
		size_t bodyLen = 0;
		{
			const size_t pos = req.find("Content-Length:");
			if (pos != std::string::npos)
				bodyLen = static_cast<size_t>(std::atoll(req.c_str() + pos + 15));
		}
		const size_t headerEnd = req.find("\r\n\r\n");
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
			// handler 异常不得逃逸出服务线程（否则 std::terminate 杀死整个进程）
			status = 500;
			respBody = std::string(R"({"error":{"code":"server_error","message":")") + e.what() + "\"}";
		}
		const char* statusText = (status == 200) ? "OK" : (status == 404) ? "Not Found" : "Internal Server Error";
		const std::string resp =
			"HTTP/1.1 " + std::to_string(status) + " " + statusText + "\r\n"
			"Content-Type: application/json\r\n"
			"Content-Length: " + std::to_string(respBody.size()) + "\r\n"
			"Connection: close\r\n\r\n" + respBody;
		c.sendBytes(resp.data(), static_cast<int>(resp.size()));
		c.close();
	}

	int _port = -1;
	std::atomic<bool> _stop{false};
	std::thread _thread;
	Poco::Net::ServerSocket _socket;
};
