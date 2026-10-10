#pragma once

#include "Node.h"
#include "Value.h"

#include <chrono>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace DC {

// 类型擦除的引擎级运行时句柄；每个 engineType 一个，由该类型全部实例共享持有。
// release 只移除注册表条目，销毁发生在最后一个共享句柄释放时。
class EngineCore {
public:
	EngineCore() = default;

	template <typename T>
	EngineCore(std::shared_ptr<T> core)
		: _core(std::move(core)) {}

	~EngineCore();

	// 移动后源不再触发释放钩子
	EngineCore(EngineCore&& other) noexcept
		: _core(std::move(other._core)), _desc(other._desc) {
		other._desc = nullptr;
	}
	EngineCore& operator=(EngineCore&&) = delete;
	EngineCore(const EngineCore&) = delete;
	EngineCore& operator=(const EngineCore&) = delete;

	void* get() {
		return _core.get();
	}
	const void* get() const {
		return _core.get();
	}

	void setDescriptor(const EngineDescriptor* desc) {
		_desc = desc;
	}

	explicit operator bool() const {
		return _core != nullptr;
	}

private:
	std::shared_ptr<void> _core;
	const EngineDescriptor* _desc = nullptr;
};

/// @brief 引擎核心共享句柄：注册表缓存与全部实例共同持有。
using EngineCoreHandle = std::shared_ptr<EngineCore>;

// 类型擦除的模型级运行时句柄；节点经 EngineAdapter 持有共享句柄。
// release 只移除注册表条目，销毁发生在最后一个共享句柄释放时。
class EngineInstance {
public:
	EngineInstance() = default;

	template <typename T>
	EngineInstance(std::shared_ptr<T> engine)
		: _engine(std::move(engine)) {}

	~EngineInstance();

	// 移动后源不再触发释放钩子
	EngineInstance(EngineInstance&& other) noexcept
		: _core(std::move(other._core)), _engine(std::move(other._engine)), _desc(other._desc) {
		other._desc = nullptr;
	}
	EngineInstance& operator=(EngineInstance&&) = delete;
	EngineInstance(const EngineInstance&) = delete;
	EngineInstance& operator=(const EngineInstance&) = delete;

	void* get() {
		return _engine.get();
	}
	const void* get() const {
		return _engine.get();
	}

	const EngineDescriptor* descriptor() const {
		return _desc;
	}

	void setDescriptor(const EngineDescriptor* desc) {
		_desc = desc;
	}

	/// @brief 绑定所属引擎核心；核心存活期须覆盖本实例。
	void attachCore(EngineCoreHandle core) {
		_core = std::move(core);
	}

	/// @brief 所属引擎核心句柄；未绑定时为空。
	const EngineCoreHandle& core() const {
		return _core;
	}

	explicit operator bool() const {
		return _engine != nullptr;
	}

private:
	// 成员序即析构逆序：模型级对象先于核心引用销毁。
	EngineCoreHandle _core;
	std::shared_ptr<void> _engine;
	const EngineDescriptor* _desc = nullptr;
};

/// @brief 引擎实例共享句柄。
using EngineHandle = std::shared_ptr<EngineInstance>;

// 注册一个引擎所需的全部信息；全部钩子均可空。
// 两层生命周期：引擎核心每 engineType 一次，模型实例每 engineType:modelPath 一次。
struct EngineDescriptor {
	std::string engineType;

	TensorConverter converter;

	/// 节点工厂：createNode 时收集 NodeFactoryParams 并调用。
	NodeFactory factory;

	/// 引擎级初始化，每 engineType 一次；可空，返回空核心视同失败。
	std::function<EngineCore()> createEngineCore;

	/// 模型级加载，每 engineType:modelPath 一次；返回空实例视同失败。
	std::function<EngineInstance(const EngineCore& core, const std::string& modelPath)> loadModel;

	/// 从引擎实例推导输入端口列表；可空。
	std::function<std::vector<Node::Port>(const EngineInstance&)> getInputPorts;

	/// 从引擎实例推导输出端口列表；可空。
	std::function<std::vector<Node::Port>(const EngineInstance&)> getOutputPorts;

	// 执行相位协议，顺序即契约：成功路径 preRun → RunFn → synchronize → postRun，
	// 任一相位失败转 onError 并跳过后续相位。
	struct ExecutionPhases {
		/// 发射前引擎级准备；此时拿不到 task 输入。
		std::function<void(void* engine)> preRun;

		/// 等待异步计算完成；同步引擎留空。
		std::function<void(void* engine)> synchronize;

		/// synchronize 成功后的后处理，仅成功路径调用。
		/// @note 缺 synchronize 而设 postRun 会产生 device 数据竞态。
		std::function<void(void* engine, Node::RunContext& ctx)> postRun;

		/// 任一相位失败后的引擎状态复位；尽力而为，自身异常被吞。
		std::function<void(void* engine)> onError;
	};
	ExecutionPhases phases;

	/// 模型实例最后一个共享句柄析构时调用一次；nullptr 时退化为默认析构。
	std::function<void(void* engine)> releaseModel;

	/// 引擎核心最后一个共享句柄析构时调用一次；nullptr 时退化为默认析构。
	std::function<void(void* core)> releaseEngineCore;
};

class EngineRegistry {
public:
	static EngineRegistry& instance();

	bool registerEngine(const EngineDescriptor& desc);

	/// @brief 从已注册引擎创建节点。
	std::unique_ptr<Node> createNode(const std::string& engineType, const std::string& nodeName,
									 const void* engineConfig = nullptr) const;

	/// @brief 直接创建节点，无需注册引擎。
	std::unique_ptr<Node> createNode(const std::string& nodeName, Node::Schema schema, Node::RunFn fn) const;

	/// @brief 从已注册引擎与模型路径创建节点；自动加载并缓存实例，Schema 由实例推导。
	std::unique_ptr<Node> createNode(const std::string& engineType, const std::string& nodeName,
									 const std::string& modelPath);

	/// @brief 物化延迟加载节点：只建节点，不创建引擎实例；
	///        实例由调用方经 getOrCreateEngine + Node::bindEngine 绑定。
	/// @return nullptr 若引擎未注册或未注册工厂
	std::unique_ptr<Node> createLazyNode(const std::string& engineType, const std::string& nodeName,
										 Node::Schema schema) const;

	/// 获取或创建引擎核心，每 engineType 一次；无钩子时合成空核心。
	EngineCoreHandle getOrCreateEngineCore(const std::string& engineType);

	/// 获取或创建引擎实例，按 engineType:modelPath 缓存；同 key 并发只加载一次。
	EngineHandle getOrCreateEngine(const std::string& engineType, const std::string& modelPath);

	/// 移除核心缓存条目；实际释放在最后一个共享句柄析构时。
	void releaseEngineCore(const std::string& engineType);

	/// 移除实例缓存条目；实际释放在最后一个共享句柄析构时。
	void releaseEngine(const std::string& engineType, const std::string& modelPath);

	/// 移除全部引擎缓存条目。
	void releaseAllEngines();

	const EngineDescriptor* find(const std::string& engineType) const;
	bool hasEngine(const std::string& engineType) const;
	std::vector<std::string> engineTypes() const;

	/// @brief 注册算子节点类型。
	/// @return true = 注册成功；false = 已存在同名算子
	bool registerOperator(const std::string& operatorName, Node::Schema schema, Node::RunFn fn);

	/// @brief 从已注册算子创建节点。
	std::unique_ptr<Node> createOperator(const std::string& operatorName, const std::string& nodeName) const;

private:
	EngineRegistry() = default;

	static std::string _makeEngineKey(const std::string& engineType, const std::string& modelPath);

	/// @brief LRU 驱逐至上限内；调用方须持 _mutex，返回被逐句柄供锁外析构，keepKey 不逐。
	std::vector<EngineHandle> _evictOverflowLocked(const std::string& keepKey);

	// 容器访问互斥；用户回调一律在锁外调用。
	mutable std::mutex _mutex;

	/// 引擎槽位，实例层与核心层共用：就绪句柄、single-flight 创建中条目、LRU 访问序。
	template <typename HandleT>
	struct EngineSlot {
		HandleT ready;
		std::shared_future<HandleT> loading;
		std::chrono::steady_clock::time_point lastAccess{};
	};

	std::unordered_map<std::string, EngineDescriptor> _engines;
	std::unordered_map<std::string, EngineSlot<EngineHandle>> _engineInstances;
	std::unordered_map<std::string, EngineSlot<EngineCoreHandle>> _engineCores;
};

// 节点工厂辅助模板

template <typename F>
NodeFactory makeNodeFactory(std::string engineType, Node::Schema schema, F&& fn) {
	return [engineType = std::move(engineType), schema = std::move(schema),
			fn = std::forward<F>(fn)](const NodeFactoryParams& p) -> std::unique_ptr<Node> {
		return std::make_unique<Node>(engineType, p.nodeName, schema, fn, ResourceClass::Compute);
	};
}

// 带配置版本：engineConfig 拷贝为值，fn 接收 const C&
template <typename C, typename F>
NodeFactory makeNodeFactory(std::string engineType, Node::Schema schema, F&& fn) {
	return [engineType = std::move(engineType), schema = std::move(schema),
			fn = std::forward<F>(fn)](const NodeFactoryParams& p) -> std::unique_ptr<Node> {
		C config = p.engineConfig ? *static_cast<const C*>(p.engineConfig) : C{};
		return std::make_unique<Node>(
			engineType, p.nodeName, schema,
			[fn, config = std::move(config)](Node::RunContext& ctx) -> Node::Result { return fn(ctx, config); },
			ResourceClass::Compute);
	};
}

// 带引擎实例版本：句柄由节点持有，实例存活期覆盖节点
template <typename F>
NodeFactory makeNodeFactoryWithEngine(std::string engineType, Node::Schema schema, F&& fn) {
	return [engineType = std::move(engineType), schema = std::move(schema),
			fn = std::forward<F>(fn)](const NodeFactoryParams& p) -> std::unique_ptr<Node> {
		auto node = std::make_unique<Node>(engineType, p.nodeName, schema, fn,
									  ResourceClass::Compute);
		if (p.engineInstance)
			node->bindEngine(p.engineInstance, p.engineInstance->descriptor());
		return node;
	};
}

} // namespace DC
