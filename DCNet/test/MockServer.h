#pragma once

// 极简 HTTP/1.1 测试服务（WinSock）：DCNet 集成测试用 Mock 远端。
// 单线程顺序处理；POST 请求体 → handler(path, body) → 响应体。
// 非 Windows 平台不提供实现（HttpTransportTest 在 WIN32 下编译）。

#include <atomic>
#include <functional>
#include <string>
#include <thread>
#include <vector>

class MockHttpServer {
public:
	/// path: 请求路径（如 /v1/infer）；返回响应体，经 status 输出 HTTP 状态码。
	using Handler = std::function<std::string(const std::string& path, const std::string& body, int& status)>;

	MockHttpServer() = default;
	~MockHttpServer() { stop(); }

	/// 绑定 127.0.0.1:0（随机端口）并启动监听线程；返回实际端口，失败返回 -1。
	int start(Handler handler);

	/// 停止监听并等待处理线程退出。
	void stop();

	int port() const { return _port; }

private:
	void run(Handler handler);

	int _port = -1;
	std::atomic<bool> _stop{false};
	std::thread _thread;
	void* _listen = nullptr; // SOCKET
};

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>

#include <cstdlib>

inline int MockHttpServer::start(Handler handler) {
	WSADATA wsa;
	if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0)
		return -1;

	SOCKET s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
	if (s == INVALID_SOCKET)
		return -1;
	sockaddr_in addr{};
	addr.sin_family = AF_INET;
	addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
	addr.sin_port = 0; // 随机端口
	if (bind(s, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == SOCKET_ERROR) {
		closesocket(s);
		return -1;
	}
	sockaddr_in bound{};
	int len = sizeof(bound);
	getsockname(s, reinterpret_cast<sockaddr*>(&bound), &len);
	_port = ntohs(bound.sin_port);
	if (listen(s, 8) == SOCKET_ERROR) {
		closesocket(s);
		return -1;
	}
	_listen = reinterpret_cast<void*>(s);
	_stop = false;
	_thread = std::thread([this, handler = std::move(handler)]() mutable { run(std::move(handler)); });
	return _port;
}

inline void MockHttpServer::stop() {
	_stop = true;
	if (_listen) {
		closesocket(reinterpret_cast<SOCKET>(_listen)); // 中断 accept
		_listen = nullptr;
	}
	if (_thread.joinable())
		_thread.join();
	WSACleanup();
}

inline void MockHttpServer::run(Handler handler) {
	const SOCKET listen = reinterpret_cast<SOCKET>(_listen);
	for (;;) {
		const SOCKET c = accept(listen, nullptr, nullptr);
		if (_stop || c == INVALID_SOCKET)
			break;

		// 读取请求（头部 + body）
		std::string req;
		char buf[4096];
		int n;
		while ((n = recv(c, buf, sizeof(buf), 0)) > 0) {
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
			n = recv(c, buf, sizeof(buf), 0);
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
		send(c, resp.data(), static_cast<int>(resp.size()), 0);
		closesocket(c);
	}
	closesocket(listen);
}
#endif // _WIN32
