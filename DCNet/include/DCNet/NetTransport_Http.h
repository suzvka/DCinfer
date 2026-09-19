#pragma once

#include "NetTransport.h"

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

/// @brief 内置 HTTP transport（POCO 实现，跨平台；DESIGN.md §9 方案 C）。
///
/// 语义（对运行时同步接口，ADR-6）：
/// - connect()：解析端点 URL（Poco::URI）→ TCP 就绪探测 → 建立会话
///   （HTTPClientSession / HTTPSClientSession，keep-alive 复用）；
///   探测失败即返回归一化错误（createEngine 配置期报告）
/// - send()：POST {basePath}{requestPath}，2xx → None（响应体留待 recv 读取）；
///   非 2xx → 读取错误体并归一化（normalizeHttpResponse）
/// - recv()：读取 2xx 响应体
///
/// 实现策略：transport 内部为同步阻塞调用，无 I/O 线程——简单 HTTP/JSON 场景
/// 直接跑在 RunFn 所在 System 池线程（ADR-6 判定矩阵第一行）。
///
/// 线程安全（DESIGN.md §2.3 契约）：引擎实例按 engineType:modelPath 缓存复用，
/// 同一 transport 可被多节点并发持有，而执行互斥粒度在节点级——故本类自行
/// 把「一次 send → recv 交换」整体串行化（连接复用、交换串行）：send 领取交换权，
/// recv（或失败/异常/close）释放。交换权为标志 + 条件变量而非锁持有，
/// 避免“由另一线程解锁互斥量”的未定义行为；send/recv 应成对在同一线程调用
/// （RunFn 契约），跨线程未收尾的占用由下一次同线程 send / close / 析构回收。
/// POCO 细节不进契约：本头文件仅前置声明，TLS 会话等实现见 .cpp。
class HttpTransport : public DcNetTransport {
public:
	HttpTransport();
	~HttpTransport() override;

	NetError connect(const NetEndpoint&) override;
	NetError send(const Payload&) override;
	NetError recv(Payload&) override;
	/// @brief 端点快照：_ep 仅在 connect（持有交换权）时写入，运行期只读。
	const NetEndpoint& endpoint() const override { return _ep; }
	bool alive() const override;
	void close() override;

private:
	Payload readBody();
	void abortResponse();
	void dropSession();
	/// 重置会话与错误位（调用者必须已持有交换权）。
	void resetLocked();
	/// 领取交换权（阻塞直到无人在交换）；同一线程已有遗留占用先回收再领。
	void acquireClaim();
	/// 释放交换权（仅当本线程持有时），并唤醒等待者。
	void releaseClaimIfOwned();

	/// 交换权 RAII 收尾：作用域结束默认释放；send 成功路径 dismiss() 把占用
	/// 延续给 recv（跨调用租约）。
	struct ClaimScope {
		HttpTransport* self;
		bool release = true;
		~ClaimScope() {
			if (release)
				self->releaseClaimIfOwned();
		}
		void dismiss() { release = false; }
	};

	NetEndpoint _ep;
	mutable std::mutex _ioMutex;                    ///< 保护下列可变状态 + 交换权登记
	std::condition_variable _ioCv;                  ///< 交换权释放时唤醒等待者
	bool _claimed = false;                          ///< 一次交换进行中（connect/send→recv/close）
	std::thread::id _claimOwner{};                  ///< 交换权持有线程（同线程重入回收依据）
	std::unique_ptr<Poco::Net::HTTPClientSession> _session; ///< HTTPS 时指向 HTTPSClientSession
	std::istream* _response = nullptr;              ///< 挂起的响应流（send → recv 之间有效）
	std::string _basePath;                          ///< 端点 basePath（不含 requestPath）
	bool _useTls = false;
	bool _failed = false;
	NetError _connectError; ///< connect 失败的归一化错误（send 未连接时复现）
};

} // namespace DC::Net
