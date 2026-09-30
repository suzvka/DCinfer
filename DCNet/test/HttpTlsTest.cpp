// HttpTransport TLS 显式校验 回归测试（P1：HTTPS 不再依赖宿主全局 SSLManager 配置）
//
// 覆盖：
//   - 默认兜底上下文：宿主未初始化 SSLManager 时框架兜底 VERIFY_STRICT，
//     不受信任的自签证书 → TlsFailed（不再是"未初始化即异常/依赖全局"）
//   - 宿主自定义信任锚：initializeClient 注入信任自签 CA 的上下文 →
//     hostname 匹配（localhost）→ 正常 TLS 交换
//   - hostname 校验被强制：信任证书但 host=127.0.0.1（证书仅含 SAN=localhost）
//     → TlsFailed
// 注：Windows SChannel（NetSSL_Win）后端的证书加载 API 不同，本测试首版
// 仅覆盖 OpenSSL 后端（CI 的 DCNet job 在 Ubuntu 运行）；Windows 侧人工验证。
#if !defined(_WIN32)

#include "DCNet/NetEndpoint.h"
#include "DCNet/NetTransport_Http.h"

#include <Poco/Net/Context.h>
#include <Poco/Net/SecureServerSocket.h>
#include <Poco/Net/SecureStreamSocket.h>
#include <Poco/Net/SocketAddress.h>
#include <Poco/Net/SSLManager.h>
#include <Poco/Timespan.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>

static int g_checks = 0;
static int g_failures = 0;

#define CHECK(cond, msg)                                                                                               \
	do {                                                                                                               \
		++g_checks;                                                                                                    \
		if (!(cond)) {                                                                                                 \
			++g_failures;                                                                                              \
			std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, msg);                                                  \
		}                                                                                                              \
	} while (0)

using namespace DC::Net;

// ── 自签证书（CN=localhost, SAN=DNS:localhost）与私钥（测试专用，无敏感值）──
static const char* kCertPem = R"PEM(-----BEGIN CERTIFICATE-----
MIIDITCCAgmgAwIBAgIUSWFZRMI7usIfWbvcjKu+OEVtg3AwDQYJKoZIhvcNAQEL
BQAwFDESMBAGA1UEAwwJbG9jYWxob3N0MCAXDTI2MDkzMDEwMzk1M1oYDzIxMjYw
OTA2MTAzOTUzWjAUMRIwEAYDVQQDDAlsb2NhbGhvc3QwggEiMA0GCSqGSIb3DQEB
AQUAA4IBDwAwggEKAoIBAQCh8CV/rOWtdPWZE5jms5cYWcj0Y1KcRBYoHYK/beGj
NKE3e43SBLjpI7z/4Q26jwxkdn4SBSzi5TjY4/t8sRE3g6b4uRi69YG4pujW7FQv
V3b8kinJuDkox7rQxNDqKuJkNtXbUvKwypRQZIciVPqSM1n/+KkDvmo3brS26wOU
kgCo5xuK6/xxISlinhR4NwRZYkMebzj8AZQfx0UV96bhQwKMWoe6Gt396kOtKQ1J
qkp1UD5OAj9QBeJQymE5coiG4K9BfyMF6q5u3LR2zDKasCn5wWXLlEgDECkMkfYe
0A9kJjAR/JUpFh3PhYgCNzQquhjfKBaDYabZljvPYKXNAgMBAAGjaTBnMB0GA1Ud
DgQWBBQ8OZCZokPwkzgR+nGo68v067HvZzAfBgNVHSMEGDAWgBQ8OZCZokPwkzgR
+nGo68v067HvZzAPBgNVHRMBAf8EBTADAQH/MBQGA1UdEQQNMAuCCWxvY2FsaG9z
dDANBgkqhkiG9w0BAQsFAAOCAQEAm5YKQ0B0TXoK80LBL/HeDEnD65npMHJVHP2f
E3TmyamFUYaNNk7gDnIHXfuEdzl0HqWgEsqYEDIKZhrbvlxPgnVq1yhon9+5y9vA
u7rxvqR9PeYziRIm0eEbhVYUz6IyFkmFYAPPCIwt88wgr9/7y06nsUhqzfhF8YWZ
+OZCMoj1Avdpe3s0t5V1m9Wq4MsQRZ6MzSCKo8R8Vx5/u91LVli/lLEf88fJUuUZ
Q6TOzlX3VuEosm1+84ivFQzvnJb7QSfJLyBeAWJMy/P0mB/D41ZEqLA1UUESKtED
/elwtJdIVULlAKYcL5brjHMpkoru8/YnxjSzDIUZOqloBsLu6A==
-----END CERTIFICATE-----
)PEM";

static const char* kKeyPem = R"PEM(-----BEGIN PRIVATE KEY-----
MIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQCh8CV/rOWtdPWZ
E5jms5cYWcj0Y1KcRBYoHYK/beGjNKE3e43SBLjpI7z/4Q26jwxkdn4SBSzi5TjY
4/t8sRE3g6b4uRi69YG4pujW7FQvV3b8kinJuDkox7rQxNDqKuJkNtXbUvKwypRQ
ZIciVPqSM1n/+KkDvmo3brS26wOUkgCo5xuK6/xxISlinhR4NwRZYkMebzj8AZQf
x0UV96bhQwKMWoe6Gt396kOtKQ1Jqkp1UD5OAj9QBeJQymE5coiG4K9BfyMF6q5u
3LR2zDKasCn5wWXLlEgDECkMkfYe0A9kJjAR/JUpFh3PhYgCNzQquhjfKBaDYabZ
ljvPYKXNAgMBAAECggEAHtTHdur2oZMyjU3vXwETQ9YYTfs5B7po04Nm2MZ1XqrP
BO63niQ7BlxBCCCTihDhJaFvuEOW+63zqEujnmZh5kVg/VrUTAghBgR1MTI2hvrq
kwTLAvZZn5uDRGsscWDv0G+mQMcmoKU5HqM9HTq7qCkxuevgVe+jbmFb87WD7X2f
J1ZjRa7Uip1jmTNT8mSiURFTpF/hvIQTsIv+bbzG96ClBKpiA20z1iSLVFA/cUJC
9+e9STK0uaKrCnjfAyD/fR8afVqXGDf4PQyMif5g0p/K50ksFOzTnkRUe8d/IEkV
u55GN1YfzVkg22DgmMxNb2hjmLuFOSD8QkCtSBu5+wKBgQDQ2X/XPXT6WSkFuwvD
GMpvGN6r7NH6Eu8raQNS8/nZKBQ/uyHki2J94FNe2Fj+TbcnH+wicTp8DMWpmXdw
Q+Q9Dh9Uqhf7SSUekRTT4Hon734jyun+o88UNrSfN/LZFpH+h+cVqdufzqpz+PrS
zs6ThRYjDQh6w7NRmyfvKIONMwKBgQDGf2LyHD4e0YadXpLmSlTCMLFJqpzC5eAG
hxHuYjnH2OOn4edN/WdBoW8FEPofjSy20OtSXe8/yIARDdM+g2EAU3QgOLx3NrE4
ZvtaOy+blwugS1Z/b2WVES7D7AghmggLXi4ZF2Xclv+LlfCAyWZYabGQGGl9oxi8
cGvzFE0A/wKBgQCBuDZheHip7qs+NfmOSl2iN65G1ydszknjiqxX39Y1/WDmXNMm
YzTfvm/KH1LXUWoLURaYJgAPgNddCkdXYbPoAFeRfLy8haganj5zg6AcIfMVRDmm
whQjF/+ETXn3QL+Zeswbdo9FaVYSBnm0amOA2U7wom274sYET/yz3VQoZQKBgCnC
BLPAQ0VCeNpEWgz+WCReD/3aWY4aw+07nwcSPOuQ8huQR5O9mmpRJsTfFG9syJpR
CyBRyJIXgPGVgfols1NZOxXIOcWuiMu/xmLuDo7h0L1Q/AplCe65Jahr0C4ZdFXH
41S9+lzUmz/nNCgztkclPQh+Sjr3A64ozFzfyW9LAoGBAL60g+/qmLWdmGELyf9P
IIfsgS2S/2OhcA1K1PXFXCM23GsE++TDgOUBbO56WM6nAoEh8BhFTlKDDgVu8lxV
DNsv81ivzqq4oQicHt/pgr8at+y4Kp3kpqEn9WjnkAvI/YaYMKQ6MFzPAw72jQe8
bItzAx9uc5oJQRIPUr8u8cLO
-----END PRIVATE KEY-----
)PEM";

namespace {

void writePemFile(const std::filesystem::path& path, const char* pem) {
	std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
	ofs << pem;
}

/// 单连接 HTTPS mock 服务：SecureServerSocket + 简单回显（POST body → JSON）。
class MockHttpsServer {
public:
	bool start() {
		try {
			const auto dir = std::filesystem::temp_directory_path() / "dcnet_tls_test";
			std::filesystem::create_directories(dir);
			writePemFile(dir / "cert.pem", kCertPem);
			writePemFile(dir / "key.pem", kKeyPem);
			Poco::Net::Context::Ptr sctx(new Poco::Net::Context(
				Poco::Net::Context::SERVER_USE, (dir / "key.pem").string(),
				(dir / "cert.pem").string(), "", Poco::Net::Context::VERIFY_NONE));
			_socket = std::make_unique<Poco::Net::SecureServerSocket>(
				Poco::Net::SocketAddress("127.0.0.1", 0), 16, sctx);
			_port = static_cast<int>(_socket->address().port());
			_stop = false;
			_thread = std::thread([this] { run(); });
			return _port > 0;
		} catch (...) {
			return false;
		}
	}

	void stop() {
		_stop = true;
		if (_thread.joinable()) {
			try {
				_socket->close();
			} catch (...) {
			}
			_thread.join();
		}
	}

	int port() const { return _port; }

private:
	void run() {
		while (!_stop) {
			Poco::Net::StreamSocket c;
			try {
				c = _socket->acceptConnection();
			} catch (...) {
				break; // socket 已关闭（stop）
			}
			serve(c);
		}
	}

	static void serve(Poco::Net::StreamSocket& c) {
		try {
			c.setReceiveTimeout(Poco::Timespan(5, 0));
			c.setSendTimeout(Poco::Timespan(5, 0));
			std::string req;
			char buf[4096];
			int n;
			while ((n = c.receiveBytes(buf, sizeof(buf))) > 0) {
				req.append(buf, static_cast<size_t>(n));
				if (req.find("\r\n\r\n") != std::string::npos)
					break;
			}
			const size_t headerEnd = req.find("\r\n\r\n");
			const std::string body =
				(headerEnd == std::string::npos) ? std::string() : req.substr(headerEnd + 4);
			// 请求体按 Content-Length 读全（简化：codec body 单帧即完整）
			std::string cl = "0";
			if (auto pos = req.find("Content-Length:"); pos != std::string::npos) {
				cl = req.substr(pos + 15, req.find("\r\n", pos) - pos - 15);
			}
			const std::string resp =
				"HTTP/1.1 200 OK\r\n"
				"Content-Type: application/json\r\n"
				"Content-Length: " + std::to_string(body.size()) + "\r\n"
				"Connection: close\r\n\r\n" + body;
			std::string out = resp;
			(void)cl;
			std::size_t sent = 0;
			while (sent < out.size()) {
				const int w = c.sendBytes(out.data() + sent, static_cast<int>(out.size() - sent));
				if (w <= 0)
					break;
				sent += static_cast<std::size_t>(w);
			}
			c.close();
		} catch (...) {
			// TLS 握手失败/连接中断：单连接静默收尾
		}
	}

	int _port = -1;
	std::atomic<bool> _stop{false};
	std::thread _thread;
	std::unique_ptr<Poco::Net::SecureServerSocket> _socket;
};

} // namespace

int main() {
	MockHttpsServer server;
	if (!server.start()) {
		std::printf("SKIP: HTTPS mock server failed to start\n");
		return 0;
	}

	// ── Test 1: 默认兜底上下文拒绝不受信证书 ──
	// 宿主未初始化 SSLManager → 框架兜底 VERIFY_STRICT（系统 CA，不含自签
	// 证书）→ 握手期证书校验失败 → 归一化 TlsFailed
	{
		HttpTransport t;
		auto ep = NetEndpoint::parse("https://localhost:" + std::to_string(server.port()) + "/v1");
		ep.connectTimeout = std::chrono::milliseconds(3000);
		ep.requestTimeout = std::chrono::milliseconds(3000);
		const auto err = t.connect(ep);
		CHECK(!err.ok(), "untrusted self-signed cert must be rejected by default strict context");
		CHECK(err.category == NetErrorCategory::Unreachable,
			  "certificate rejection must normalize to the TLS family (Unreachable)");
	}

	// ── Test 2: 宿主自定义信任锚（信任自签 CA）→ hostname 匹配 → 成功 ──
	{
		const auto dir = std::filesystem::temp_directory_path() / "dcnet_tls_test";
		Poco::Net::Context::Ptr ctx(new Poco::Net::Context(
			Poco::Net::Context::CLIENT_USE, "", "", dir.string(),
			Poco::Net::Context::VERIFY_STRICT, 9, false));
		Poco::Net::SSLManager::instance().initializeClient(nullptr, nullptr, ctx);

		HttpTransport t;
		auto ep = NetEndpoint::parse("https://localhost:" + std::to_string(server.port()) + "/v1");
		ep.connectTimeout = std::chrono::milliseconds(3000);
		ep.requestTimeout = std::chrono::milliseconds(3000);
		const auto err = t.connect(ep);
		CHECK(err.ok(), "trusted self-signed CA + matching hostname (localhost) must connect");
		if (err.ok()) {
			CHECK(t.send(R"({"probe":true})").ok(), "send over TLS");
			Payload body;
			CHECK(t.recv(body).ok(), "recv over TLS");
			CHECK(body.find(R"({"probe":true})") != std::string::npos, "TLS payload round-trip");
		}
	}

	// ── Test 3: hostname 校验被强制（信任证书但 host 与 SAN 不匹配）──
	{
		HttpTransport t;
		// 证书仅含 CN=localhost / SAN=DNS:localhost；host=127.0.0.1 → 校验失败
		auto ep = NetEndpoint::parse("https://127.0.0.1:" + std::to_string(server.port()) + "/v1");
		ep.connectTimeout = std::chrono::milliseconds(3000);
		ep.requestTimeout = std::chrono::milliseconds(3000);
		const auto err = t.connect(ep);
		CHECK(!err.ok(), "hostname mismatch (127.0.0.1 vs SAN=localhost) must be rejected");
	}

	server.stop();
	std::printf("HttpTlsTest: %d checks, %d failures\n", g_checks, g_failures);
	return g_failures == 0 ? 0 : 1;
}

#else // _WIN32

#include <cstdio>
int main() {
	// Windows SChannel（NetSSL_Win）后端的证书加载 API 不同，首版跳过
	//（见修复计划风险 2）；CI 的 DCNet job 在 Ubuntu/OpenSSL 运行本测试
	std::printf("SKIP: TLS test targets the OpenSSL backend only\n");
	return 0;
}

#endif
