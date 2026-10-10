// HttpTransport TLS 显式校验回归测试。
//
// 语义前提：connect() 仅做 TCP 就绪探测，TLS 握手延迟到首次 send，故所有证书
// 校验断言必须在 send/recv 层验证，在 connect 层断言会产生假阳性。
//
// 覆盖：默认兜底上下文拒绝不受信自签证书；宿主注入信任锚后正常交换；
// hostname 校验强制生效，SAN=localhost 对 127.0.0.1 拒绝。
//
// 服务端基础设施注记，均曾为真实缺陷：监听 [::]:0 的双栈解析，Ubuntu 上
// localhost 可能走 ::1；accept 异常不得退出服务循环；caLocation 必须为文件
// 路径，OpenSSL 目录模式需 hash 命名。
// Windows uses native SChannel; trust stays in Context::addTrustedCert (memory only).

#include "DCNet/NetEndpoint.h"
#include "DCNet/NetError.h"
#include "DCNet/NetTransport_Http.h"

#include <Poco/Net/Context.h>
#include <Poco/Net/SecureServerSocket.h>
#include <Poco/Net/SecureStreamSocket.h>
#include <Poco/Net/SocketAddress.h>
#include <Poco/Net/SSLManager.h>
#include <Poco/Timespan.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>

#include <Poco/Net/X509Certificate.h>
#include <Poco/UUIDGenerator.h>
#include <atomic>
#include <memory>
#include <stdexcept>
#include <vector>
#if defined(_WIN32)
#include <Poco/Delegate.h>
#include <Poco/Net/VerificationErrorArgs.h>
#pragma comment(lib, "crypt32.lib")
#pragma comment(lib, "advapi32.lib")
#endif

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

// 自签证书 CN=localhost, SAN=DNS:localhost 与私钥：测试专用。
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

#if defined(_WIN32)
// Observe POCO's verification event without changing its decision: transport errors
// intentionally redact diagnostics, so public text cannot prove a certificate rejection.
class VerificationObserver {
public:
	VerificationObserver() {
		Poco::Net::SSLManager::instance().ClientVerificationError +=
			Poco::delegate(this, &VerificationObserver::onError);
	}
	~VerificationObserver() {
		Poco::Net::SSLManager::instance().ClientVerificationError -=
			Poco::delegate(this, &VerificationObserver::onError);
	}
	bool untrusted = false;
	bool hostname = false;
private:
	void onError(const void*, Poco::Net::VerificationErrorArgs& args) {
		untrusted = untrusted || args.errorMessage() == "Certificate Authority not trusted";
		hostname = hostname || args.errorMessage() == "The certificate host names do not match the server host name";
		// Deliberately do not setIgnoreError: this is an observer, never a bypass.
	}
};

// POCO's PFX loader persists private keys (no PKCS12_NO_PERSIST_KEY): record and
// delete only that fixture-owned key after release; certificates/trusts stay in memory.
class NativeTlsFixture {
public:
	NativeTlsFixture() {
		const auto id = Poco::UUIDGenerator::defaultGenerator().createRandom().toString();
		_dir = std::filesystem::temp_directory_path() / ("dcnet_tls_" + id);
		_keyName = L"dcnet_tls_" + std::wstring(id.begin(), id.end());
	}
	~NativeTlsFixture() {
		if (_cert) CertFreeCertificateContext(_cert);
		if (_key) CryptDestroyKey(_key);
		if (_provider) CryptReleaseContext(_provider, 0);
		if (!_importedName.empty() && _importedName != _keyName)
			deleteKey(_importedName, _importedProvider, _importedType);
		if (_createdKey) deleteKey(_keyName, MS_ENH_RSA_AES_PROV_W, PROV_RSA_AES);
		std::error_code ec;
		// Unique directory created by this fixture; never remove a shared temp path.
		if (_createdDir) std::filesystem::remove_all(_dir, ec);
		CHECK(!ec, "remove native TLS fixture directory");
	}
	void create() {
		_createdDir = std::filesystem::create_directory(_dir);
		require(_createdDir, "create unique fixture directory");
		require(CryptAcquireContextW(&_provider, _keyName.c_str(), MS_ENH_RSA_AES_PROV_W,
			PROV_RSA_AES, CRYPT_NEWKEYSET), "create fixture key container");
		_createdKey = true;
		require(CryptGenKey(_provider, AT_KEYEXCHANGE, (2048 << 16) | CRYPT_EXPORTABLE, &_key),
			"generate RSA key");
		DWORD size = 0;
		require(CertStrToNameW(X509_ASN_ENCODING, L"CN=localhost", CERT_X500_NAME_STR,
			nullptr, nullptr, &size, nullptr), "size subject");
		std::vector<BYTE> subject(size);
		require(CertStrToNameW(X509_ASN_ENCODING, L"CN=localhost", CERT_X500_NAME_STR,
			nullptr, subject.data(), &size, nullptr), "encode subject");
		CERT_NAME_BLOB name{size, subject.data()};
		CERT_ALT_NAME_ENTRY dns{};
		dns.dwAltNameChoice = CERT_ALT_NAME_DNS_NAME;
		dns.pwszDNSName = const_cast<wchar_t*>(L"localhost");
		CERT_ALT_NAME_INFO san{1, &dns};
		CERT_BASIC_CONSTRAINTS2_INFO ca{};
		ca.fCA = TRUE;
		auto sanBytes = encode(X509_ALTERNATE_NAME, &san);
		auto caBytes = encode(X509_BASIC_CONSTRAINTS2, &ca);
		CERT_EXTENSION extensions[] = {
			{const_cast<char*>(szOID_SUBJECT_ALT_NAME2), FALSE,
				{static_cast<DWORD>(sanBytes.size()), sanBytes.data()}},
			{const_cast<char*>(szOID_BASIC_CONSTRAINTS2), TRUE,
				{static_cast<DWORD>(caBytes.size()), caBytes.data()}}
		};
		CERT_EXTENSIONS ext{2, extensions};
		CRYPT_KEY_PROV_INFO keyInfo{};
		keyInfo.pwszContainerName = const_cast<wchar_t*>(_keyName.c_str());
		keyInfo.pwszProvName = const_cast<wchar_t*>(MS_ENH_RSA_AES_PROV_W);
		keyInfo.dwProvType = PROV_RSA_AES;
		keyInfo.dwKeySpec = AT_KEYEXCHANGE;
		CRYPT_ALGORITHM_IDENTIFIER algorithm{};
		algorithm.pszObjId = const_cast<char*>(szOID_RSA_SHA256RSA);
		FILETIME now;
		GetSystemTimeAsFileTime(&now);
		ULARGE_INTEGER ticks{};
		ticks.LowPart = now.dwLowDateTime; ticks.HighPart = now.dwHighDateTime;
		const auto center = ticks.QuadPart;
		SYSTEMTIME from{}, until{};
		ticks.QuadPart = center - 24ULL * 60 * 60 * 10000000;
		FILETIME ft{ticks.LowPart, ticks.HighPart};
		require(FileTimeToSystemTime(&ft, &from), "certificate start time");
		ticks.QuadPart = center + 7ULL * 24 * 60 * 60 * 10000000;
		ft = {ticks.LowPart, ticks.HighPart};
		require(FileTimeToSystemTime(&ft, &until), "certificate end time");
		_cert = CertCreateSelfSignCertificate(_provider, &name, 0, &keyInfo,
			&algorithm, &from, &until, &ext);
		require(_cert != nullptr, "create localhost self-signed CA");
		HCERTSTORE store = CertOpenStore(CERT_STORE_PROV_MEMORY, 0, 0, 0, nullptr);
		require(store != nullptr, "create export memory store");
		try {
			require(CertAddCertificateContextToStore(store, _cert, CERT_STORE_ADD_ALWAYS, nullptr),
				"add export certificate");
			CRYPT_DATA_BLOB blob{};
			require(PFXExportCertStoreEx(store, &blob, L"", nullptr, EXPORT_PRIVATE_KEYS | REPORT_NOT_ABLE_TO_EXPORT_PRIVATE_KEY), "size PFX");
			std::vector<BYTE> pfx(blob.cbData);
			blob.pbData = pfx.data();
			require(PFXExportCertStoreEx(store, &blob, L"", nullptr, EXPORT_PRIVATE_KEYS | REPORT_NOT_ABLE_TO_EXPORT_PRIVATE_KEY), "export PFX");
			std::ofstream file(pfxPath(), std::ios::binary);
			file.write(reinterpret_cast<const char*>(pfx.data()), blob.cbData);
			require(file.good(), "write PFX fixture");
		} catch (...) { CertCloseStore(store, 0); throw; }
		CertCloseStore(store, 0);
	}
	std::filesystem::path pfxPath() const { return _dir / "server.pfx"; }
	Poco::Net::X509Certificate certificate() const { return Poco::Net::X509Certificate(_cert, true); }
	void recordImportedKey(PCCERT_CONTEXT cert) {
		DWORD size = 0;
		require(CertGetCertificateContextProperty(cert, CERT_KEY_PROV_INFO_PROP_ID, nullptr, &size), "size imported key info");
		std::vector<BYTE> bytes(size);
		require(CertGetCertificateContextProperty(cert, CERT_KEY_PROV_INFO_PROP_ID, bytes.data(), &size), "read imported key info");
		const auto* info = reinterpret_cast<const CRYPT_KEY_PROV_INFO*>(bytes.data());
		_importedName = info->pwszContainerName;
		_importedProvider = info->pwszProvName;
		_importedType = info->dwProvType;
		require(!(info->dwFlags & CRYPT_MACHINE_KEYSET) && _importedType != 0,
			"fixture import must use user-scoped CryptoAPI key");
	}
private:
	static void require(bool ok, const char* what) {
		if (!ok) throw std::runtime_error(std::string(what) + " (Win32=" + std::to_string(GetLastError()) + ")");
	}
	static std::vector<BYTE> encode(LPCSTR type, const void* value) {
		DWORD size = 0;
		require(CryptEncodeObjectEx(X509_ASN_ENCODING, type, value, 0, nullptr, nullptr, &size), "size extension");
		std::vector<BYTE> bytes(size);
		require(CryptEncodeObjectEx(X509_ASN_ENCODING, type, value, 0, nullptr, bytes.data(), &size), "encode extension");
		return bytes;
	}
	static void deleteKey(const std::wstring& name, const std::wstring& provider, DWORD type) {
		HCRYPTPROV unused = 0;
		CHECK(CryptAcquireContextW(&unused, name.c_str(), provider.c_str(), type, CRYPT_DELETEKEYSET),
			"delete fixture-owned private key container");
	}
	std::filesystem::path _dir;
	std::wstring _keyName, _importedName, _importedProvider;
	DWORD _importedType = 0;
	HCRYPTPROV _provider = 0;
	HCRYPTKEY _key = 0;
	PCCERT_CONTEXT _cert = nullptr;
	bool _createdKey = false;
	bool _createdDir = false;
};
#endif

/// 单连接 HTTPS mock 服务：SecureServerSocket 加回显，POST body 转 JSON。
class MockHttpsServer {
public:
	bool start() {
		try {
#if defined(_WIN32)
			_fixture.create();
			Poco::Net::Context::Ptr sctx(new Poco::Net::Context(
				Poco::Net::Context::TLS_SERVER_USE, _fixture.pfxPath().string(),
				Poco::Net::Context::VERIFY_NONE,
				Poco::Net::Context::OPT_LOAD_CERT_FROM_FILE | Poco::Net::Context::OPT_USE_STRONG_CRYPTO));
			const auto cert = sctx->certificate();
			_fixture.recordImportedKey(cert.system());
			sctx->requireMinimumProtocol(Poco::Net::Context::PROTO_TLSV1_2);
#else
			const auto dir = std::filesystem::temp_directory_path() / "dcnet_tls_test";
			std::filesystem::create_directories(dir);
			writePemFile(dir / "cert.pem", kCertPem);
			writePemFile(dir / "key.pem", kKeyPem);
			Poco::Net::Context::Ptr sctx(new Poco::Net::Context(
				Poco::Net::Context::SERVER_USE, (dir / "key.pem").string(),
				(dir / "cert.pem").string(), "", Poco::Net::Context::VERIFY_NONE));
#endif
			// Explicit dual-stack binding is required on Windows as well: localhost
			// may resolve to either family. Keep all fixture traffic on loopback.
			_socket = std::make_unique<Poco::Net::SecureServerSocket>(sctx);
			_socket->bind6(Poco::Net::SocketAddress("[::]:0"), true, false);
			_socket->listen(16);
			_port = static_cast<int>(_socket->address().port());
			_stop = false;
			_thread = std::thread([this] { run(); });
			return _port > 0;
		} catch (const std::exception& e) {
			std::printf("HTTPS fixture setup failed: %s\n", e.what());
			return false;
		}
	}

	~MockHttpsServer() { stop(); }
#if defined(_WIN32)
	Poco::Net::X509Certificate certificate() const { return _fixture.certificate(); }
#endif

	void stop() {
		_stop = true;
		if (_thread.joinable()) _thread.join();
		_socket.reset();
	}

	int port() const { return _port; }
	std::size_t receivedBytes() const { return _receivedBytes.load(); }

private:
	void run() {
		while (!_stop) {
			Poco::Net::StreamSocket c;
			try {
				if (!_socket->poll(Poco::Timespan(100000), Poco::Net::Socket::SELECT_READ))
					continue;
				c = _socket->acceptConnection();
			} catch (...) {
				// stop 主动关闭则退出；客户端握手失败或裸 TCP probe，即 connect
				// 探测先连后关，循环必须存活，否则服务器被 probe 杀死、用例误报
				if (_stop)
					break;
				continue;
			}
			serve(c);
		}
	}

	void serve(Poco::Net::StreamSocket& c) {
		try {
			c.setReceiveTimeout(Poco::Timespan(5, 0));
			c.setSendTimeout(Poco::Timespan(5, 0));
			std::string req;
			char buf[4096];
			int n;
			while ((n = c.receiveBytes(buf, sizeof(buf))) > 0) {
				_receivedBytes += static_cast<std::size_t>(n);
				req.append(buf, static_cast<size_t>(n));
				const size_t headerEnd = req.find("\r\n\r\n");
				if (headerEnd == std::string::npos)
					continue;
				// 按 Content-Length 读全请求体：TLS 分段下 body 可能晚于 header 到达
				std::size_t cl = 0;
				if (auto pos = req.find("Content-Length:"); pos != std::string::npos) {
					cl = static_cast<std::size_t>(
						std::strtoull(req.c_str() + pos + 15, nullptr, 10));
				}
				if (req.size() - (headerEnd + 4) >= cl)
					break;
			}
			const size_t headerEnd = req.find("\r\n\r\n");
			const std::string body =
				(headerEnd == std::string::npos) ? std::string() : req.substr(headerEnd + 4);
			const std::string resp =
				"HTTP/1.1 200 OK\r\n"
				"Content-Type: application/json\r\n"
				"Content-Length: " + std::to_string(body.size()) + "\r\n"
				"Connection: close\r\n\r\n" + body;
			std::size_t sent = 0;
			while (sent < resp.size()) {
				const int w = c.sendBytes(resp.data() + sent, static_cast<int>(resp.size() - sent));
				if (w <= 0)
					break;
				sent += static_cast<std::size_t>(w);
			}
			c.close();
		} catch (...) {
			// TLS 握手失败/连接中断：单连接静默收尾
		}
	}

#if defined(_WIN32)
	NativeTlsFixture _fixture; // destroyed after the server socket/context
#endif
	int _port = -1;
	std::atomic<bool> _stop{false};
	std::atomic<std::size_t> _receivedBytes{0};
	std::thread _thread;
	std::unique_ptr<Poco::Net::SecureServerSocket> _socket;
};

} // namespace

int runTests() {
#if defined(_WIN32)
	VerificationObserver verification;
#endif
	MockHttpsServer server;
	if (!server.start()) {
		std::printf("FAIL: HTTPS mock server failed to start\n");
		return 1;
	}

	const std::string localhostEp = "https://localhost:" + std::to_string(server.port()) + "/v1";

	// Test 1：默认兜底上下文拒绝不受信证书。自签证书不在系统 CA：connect 仅
	// TCP 探测成功，send 触发握手拒绝，断言安全行为而非 mode 枚举。
	{
		HttpTransport t;
		auto ep = NetEndpoint::parse(localhostEp);
		ep.connectTimeout = std::chrono::milliseconds(3000);
		ep.requestTimeout = std::chrono::milliseconds(3000);
		const auto conn = t.connect(ep);
		CHECK(conn.ok(), "connect is a TCP readiness probe and must succeed on a live server");
		if (conn.ok()) {
			const auto err = t.send(R"({"probe":true})");
			CHECK(!err.ok(), "untrusted self-signed cert must be rejected by default strict context");
			CHECK(err.category == NetErrorCategory::Unreachable,
				  "certificate rejection must normalize to the TLS family (Unreachable)");
#if defined(_WIN32)
			CHECK(verification.untrusted,
				  "untrusted rejection must emit certificate chain validation evidence");
#endif
		}
	}

	// Test 2：宿主自定义信任锚即信任自签 CA，加 hostname 匹配则成功。
	{
#if defined(_WIN32)
		// VERIFY_RELAXED still checks chain and hostname but selects POCO's manual
		// validation path consulting addTrustedCert (VERIFY_STRICT lets SChannel reject
		// unknown roots first). No system ROOT, no bypass, no revocation checks.
		Poco::Net::Context::Ptr ctx(new Poco::Net::Context(
			Poco::Net::Context::TLS_CLIENT_USE, "", Poco::Net::Context::VERIFY_RELAXED,
			Poco::Net::Context::OPT_USE_STRONG_CRYPTO));
		ctx->addTrustedCert(server.certificate());
#else
		const auto dir = std::filesystem::temp_directory_path() / "dcnet_tls_test";
		// caLocation 必须传文件路径：OpenSSL 目录模式要求 hash 命名链接
		Poco::Net::Context::Ptr ctx(new Poco::Net::Context(
			Poco::Net::Context::CLIENT_USE, "", "", (dir / "cert.pem").string(),
			Poco::Net::Context::VERIFY_STRICT, 9, false));
#endif
		Poco::Net::SSLManager::instance().initializeClient(nullptr, nullptr, ctx);

		HttpTransport t;
		auto ep = NetEndpoint::parse(localhostEp);
		ep.connectTimeout = std::chrono::milliseconds(3000);
		ep.requestTimeout = std::chrono::milliseconds(3000);
		const auto conn = t.connect(ep);
		CHECK(conn.ok(), "trusted context: connect (TCP probe) must succeed");
		if (conn.ok()) {
			CHECK(t.send(R"({"probe":true})").ok(), "send over TLS (handshake with trusted self-signed CA)");
			Payload body;
			CHECK(t.recv(body).ok(), "recv over TLS");
			CHECK(body.find(R"({"probe":true})") != std::string::npos, "TLS payload round-trip");
		}
	}

	// Test 3：hostname 校验被强制。沿用 Test 2 的信任锚上下文：证书受信，但
	// host=127.0.0.1 与 SAN=DNS:localhost 不匹配 → send 时握手失败。
	const auto bytesBeforeMismatch = server.receivedBytes();
	{
		HttpTransport t;
		auto ep = NetEndpoint::parse("https://127.0.0.1:" + std::to_string(server.port()) + "/v1");
		ep.connectTimeout = std::chrono::milliseconds(3000);
		ep.requestTimeout = std::chrono::milliseconds(3000);
		const auto conn = t.connect(ep);
		CHECK(conn.ok(), "connect is a TCP readiness probe and must succeed regardless of TLS");
		if (conn.ok()) {
			const auto err = t.send(R"({"probe":true})");
			CHECK(!err.ok(), "hostname mismatch (127.0.0.1 vs SAN=localhost) must be rejected");
			CHECK(err.category == NetErrorCategory::Unreachable,
				  "hostname mismatch must normalize to the TLS family (Unreachable)");

		}
	}

	server.stop();
	CHECK(server.receivedBytes() == bytesBeforeMismatch,
		  "hostname rejection must occur before sending HTTP headers or body");
	return g_failures == 0 ? 0 : 1;
}

int main() {
	int result = 1;
	try {
		result = runTests();
	} catch (const std::exception& e) {
		std::printf("FAIL: TLS fixture/test exception: %s\n", e.what());
		++g_failures;
	}
#if defined(_WIN32)
	Poco::Net::SSLManager::instance().initializeClient(nullptr, nullptr, nullptr);
#endif
	// runTests has returned: fixture cleanup failures participate in the process result.
	std::printf("HttpTlsTest: %d checks, %d failures\n", g_checks, g_failures);
	return result == 0 && g_failures == 0 ? 0 : 1;
}
