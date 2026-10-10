#pragma once

#include "NetTransport.h"

#include <atomic>
#include <condition_variable>
#include <iosfwd>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace Poco::Net {
class HTTPClientSession;
}

namespace DC::Net {

/// 内置 HTTP transport（POCO 实现）：connect 探测 TCP 就绪；send POST
/// {basePath}{requestPath}，非 2xx 读错误体归一化（不跟随 3xx）；recv 读取
/// 2xx 响应体，超 maxResponseBody 按错误归一化。同步阻塞，无 I/O 线程。
///
/// 线程安全：同一 transport 可被多节点并发持有，故把「一次 send → recv 交换」
/// 整体串行化——send 领取交换权，recv（或失败 / close）释放；交换权用标志 +
/// 条件变量而非锁持有，避免“由另一线程解锁互斥量”的未定义行为。send / recv
/// 应成对在同一线程调用，跨线程未收尾的占用由下一次同线程 send / close / 析构回收。
class HttpTransport : public DcNetTransport {
public:
	HttpTransport();
	~HttpTransport() override;

	NetError connect(const NetEndpoint&) override;
	NetError send(const Payload&) override;
	NetError recv(Payload&) override;
	/// 端点快照：_ep 仅在 connect 时写入，运行期只读。
	const NetEndpoint& endpoint() const override { return _ep; }
	bool alive() const override;
	void close() override;

private:
	/// 分块读取响应体（limit = 0 不限制）；truncated 置位表示达上限被截断
	/// （2xx 成功体超限按错误处理，非 2xx 错误体允许截断）。
	Payload readBody(std::istream& rs, size_t limit, bool* truncated);
	void abortResponse();
	void dropSession();
	/// 重置会话与错误位（须已持有交换权）。
	void resetLocked();
	/// 领取交换权（阻塞直到无人在交换；同线程遗留占用先回收）。
	void acquireClaim();
	/// 释放交换权（仅当本线程持有时）并唤醒等待者。
	void finishCall(bool releaseClaim);

	/// 交换权 RAII 收尾：默认释放；send 成功路径 dismiss() 延续给 recv。
	struct ClaimScope {
		HttpTransport* self;
		bool release = true;
		~ClaimScope() {
			self->finishCall(release);
		}
		void dismiss() { release = false; }
	};

	NetEndpoint _ep;
	mutable std::mutex _ioMutex;                    ///< 保护下列可变状态 + 交换权登记
	std::condition_variable _ioCv;                  ///< 交换权释放时唤醒等待者
	bool _closing = false;
	bool _callActive = false;
	bool _claimed = false;                          ///< 一次交换进行中（connect/send→recv/close）
	std::thread::id _claimOwner{};                  ///< 交换权持有线程（同线程重入回收依据）
	/// 会话：shared_ptr 保活——交换中途被 close 强收时，副本保活至读取结束。
	std::shared_ptr<Poco::Net::HTTPClientSession> _session; ///< HTTPS 时指向 HTTPSClientSession
	/// 挂起的响应流（send → recv 之间有效；仅交换权持有期间访问）。
	std::istream* _response = nullptr;
	long long _responseLength = -1;
	std::string _basePath;                          ///< 端点 basePath（不含 requestPath）
	bool _useTls = false;
	std::atomic<bool> _failed{false};
	NetError _connectError; ///< connect 失败的归一化错误（send 未连接时复现）
};

} // namespace DC::Net
