#pragma once

#include "Node.h"
#include "Value.h"

#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace DC {

// ── 引擎实例：类型擦除的运行时引擎句柄 ──
// 封装引擎运行时对象（Ort::Session / nvinfer1::ICudaEngine / 自定义对象）
// 基于 shared_ptr<void> 实现类型擦除，EngineDescriptor 充当虚表
//
// 生命周期（EngineHandle 共享所有权）：注册表缓存实例的共享句柄，
// 节点经 EngineAdapter 同样持有句柄；releaseEngine/releaseAllEngines
// 仅移除注册表条目，实际销毁（含 releaseEngine 钩子）发生在最后一个
// 共享句柄释放时——仍持有句柄的节点安全存活，消除悬空引用。
class EngineInstance {
public:
	EngineInstance() = default;

	template <typename T>
	EngineInstance(std::shared_ptr<T> engine, const EngineDescriptor* desc = nullptr)
		: _engine(std::move(engine)), _desc(desc) {}

	~EngineInstance();

	// 共享句柄对象：禁止拷贝；移动移交释放钩子职责（源不再触发）
	EngineInstance(EngineInstance&& other) noexcept
		: _engine(std::move(other._engine)), _desc(other._desc) {
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

	/// @brief 设置所属引擎描述符。
	/// 框架在实例缓存（getOrCreateEngine）时注入，为权威值，
	/// 覆盖构造时传入的描述符——适配器无需自行查找所属描述符。
	void setDescriptor(const EngineDescriptor* desc) {
		_desc = desc;
	}

	explicit operator bool() const {
		return _engine != nullptr;
	}

private:
	std::shared_ptr<void> _engine;
	const EngineDescriptor* _desc = nullptr;
};

/// @brief 引擎实例共享句柄：节点/注册表共同持有，引用计数决定实例销毁时机。
using EngineHandle = std::shared_ptr<EngineInstance>;

// ── 引擎描述符：注册一个引擎所需的全部信息 ──
//
// 钩子按交互语义分为四组（"调用时机归谁、是否有顺序约束"各不相同）：
//   1. 纯函数工具   —— TensorConverter，由适配器作者在 RunFn 内自行调用；
//   2. 工厂与内省   —— 框架自主调度（建图时推导 Schema、首次使用时创建实例），无顺序约束；
//   3. 执行相位协议 —— 框架按固定算法调用的模板方法，顺序即契约（见 ExecutionPhases）；
//   4. 所有权事件   —— 最后一个共享句柄析构时触发，无顺序约束。
//
// 全部钩子均可空。同步引擎可把全部执行逻辑内联在 RunFn 内、执行相位全部留空
// （范例见 DCEngines/OnnxRuntime 适配器：仅 synchronize = no-op，其余留空）。
struct EngineDescriptor {
	std::string engineType;

	// ── 纯函数工具：DC::Tensor ↔ 引擎原生张量 ──
	TensorConverter converter;

	// ── 工厂与内省（框架自主调度，无顺序约束）──

	/// 节点工厂：框架在 createNode 时收集 NodeFactoryParams 并调用
	NodeFactory factory;

	/// 从模型路径创建引擎实例（含模型加载与运行时资源分配）
	/// 系统以 modelPath 为 key 缓存实例，适配器在钩子内决定复用策略
	std::function<EngineInstance(const std::string& modelPath)> createEngine;

	/// 从引擎实例推导输入端口列表，用于自动推导 Schema（可空，工厂自行兜底）
	std::function<std::vector<Node::Port>(const EngineInstance&)> getInputPorts;

	/// 从引擎实例推导输出端口列表，用于自动推导 Schema（可空，工厂自行兜底）
	std::function<std::vector<Node::Port>(const EngineInstance&)> getOutputPorts;

	// ── 执行相位协议（顺序即契约：框架掌序、适配器填空）──
	//
	// 成功路径:  preRun → RunFn(框架驱动) → synchronize → postRun
	// 失败路径:  任一相位（preRun / RunFn / synchronize / postRun）失败
	//            → onError（后续相位跳过；onError 自身异常被吞，不传播次生异常）
	// 约束:      postRun 依赖 synchronize 已设置且已执行（device 数据可见性）
	struct ExecutionPhases {
		/// 发射前引擎级准备（warmup、与 task 数据无关的 session 配置等）。
		/// 注意：本钩子拿不到 task 输入——输入绑定发生在 RunFn 内经 TensorConverter 完成。
		/// engine 为 EngineInstance::get() 返回的原生指针。
		std::function<void(void* engine)> preRun;

		/// 等待异步计算完成（发射后、输出收集前）。同步引擎留空或 no-op。
		std::function<void(void* engine)> synchronize;

		/// synchronize 成功后的后处理（仅成功路径调用）。
		/// 典型用途：device→host 数据传输、输出格式后处理。
		/// ctx 提供完整的输入/输出槽位访问，可通过 ctx.outputRaw() 读取 GPU 输出、
		/// ctx.output() 写回 host 数据。
		/// @note 依赖 synchronize 已设置且已执行——异步引擎留空 synchronize 而设置
		///       postRun，会在 device 数据未就绪时产生竞态。
		std::function<void(void* engine, Node::RunContext& ctx)> postRun;

		/// 任一执行相位（preRun / RunFn / synchronize / postRun）失败后的
		/// 引擎状态复位（尽力而为；自身异常被吞，不传播次生异常）。
		std::function<void(void* engine)> onError;
	};
	ExecutionPhases phases;

	// ── 所有权事件（无顺序约束）──

	/// 最后一个共享句柄析构时调用一次，用于有序清理 GPU 资源
	/// 若为 nullptr，退化为 shared_ptr<void> 默认析构
	std::function<void(void* engine)> releaseEngine;
};

// ── 引擎注册表 ──
class EngineRegistry {
public:
	static EngineRegistry& instance();

	bool registerEngine(const EngineDescriptor& desc);

	// ── 接口 1：从已注册引擎创建节点 ──
	std::unique_ptr<Node> createNode(const std::string& engineType, const std::string& nodeName,
									 const void* engineConfig = nullptr) const;

	// ── 接口 2：直接创建一个节点（无需注册引擎，自动标记为 "Builtin"）──
	std::unique_ptr<Node> createNode(const std::string& nodeName, Node::Schema schema, Node::RunFn fn) const;

	// ── 接口 3：从已注册引擎 + 模型路径创建节点 ──
	// 自动调用 getOrCreateEngine（一次加载并缓存实例）→ 从实例推导 Schema → 构建节点
	// 节点经 EngineAdapter 持有 EngineInstance 共享句柄，实例生命周期由句柄引用计数管理
	std::unique_ptr<Node> createNode(const std::string& engineType, const std::string& nodeName,
									 const std::string& modelPath);

	// ── 引擎实例管理 ──

	/// 获取或创建引擎实例（以 engineType + modelPath 复合键缓存）
	/// 若未缓存则调用 EngineDescriptor::createEngine 创建
	/// 返回共享句柄：注册表与调用方共同持有，引用计数决定实例销毁时机
	EngineHandle getOrCreateEngine(const std::string& engineType, const std::string& modelPath);

	/// 移除指定引擎类型 + 模型路径的缓存条目（不销毁实例：
	/// 实际释放发生在最后一个共享句柄析构时，仍被节点持有的实例安全存活）
	void releaseEngine(const std::string& engineType, const std::string& modelPath);

	/// 移除全部引擎实例缓存条目（语义同上，逐条目移除）
	void releaseAllEngines();

	const EngineDescriptor* find(const std::string& engineType) const;
	bool hasEngine(const std::string& engineType) const;
	std::vector<std::string> engineTypes() const;

	// ── 算子注册（轻量级，DC::Tensor only，无引擎钩子）──

	/// @brief  注册一个算子节点类型
	/// @param  operatorName  算子名（如 "Broadcast", "Add"）
	/// @param  schema        输入/输出端口 Schema
	/// @param  fn            算子计算逻辑
	/// @return true 表示注册成功，false 表示已存在同名算子
	bool registerOperator(const std::string& operatorName, Node::Schema schema, Node::RunFn fn);

	/// @brief  从已注册算子创建节点（无需 engineConfig）
	std::unique_ptr<Node> createOperator(const std::string& operatorName, const std::string& nodeName) const;

private:
	EngineRegistry() = default;

	static std::string _makeEngineKey(const std::string& engineType, const std::string& modelPath);

	// 容器访问互斥：注册表支持并发建图（多个线程同时 getOrCreateEngine/createNode）
	// 用户回调（factory / createEngine / 端口推导）一律在锁外调用（single-flight）
	mutable std::mutex _mutex;

	/// 引擎槽位：ready = 已就绪实例（缓存条目）；
	/// loading = single-flight 创建中条目（同 key 并发首个创建者登记，
	/// 其余调用者经 shared_future 等待同一结果，创建回调在锁外执行）。
	struct EngineSlot {
		EngineHandle ready;
		std::shared_future<EngineHandle> loading;
	};

	std::unordered_map<std::string, EngineDescriptor> _engines;
	std::unordered_map<std::string, EngineSlot> _engineInstances;
};

// ── 节点工厂辅助模板 ──
// 无配置版本
template <typename F>
NodeFactory makeNodeFactory(std::string engineType, Node::Schema schema, F&& fn) {
	return [engineType = std::move(engineType), schema = std::move(schema),
			fn = std::forward<F>(fn)](const NodeFactoryParams& p) -> std::unique_ptr<Node> {
		return std::make_unique<Node>(engineType, p.nodeName, schema, fn, ThreadPoolAffinity::Compute);
	};
}

// 带配置版本：engineConfig 在 createNode 时拷贝为值，lambda 接收 const C&
template <typename C, typename F>
NodeFactory makeNodeFactory(std::string engineType, Node::Schema schema, F&& fn) {
	return [engineType = std::move(engineType), schema = std::move(schema),
			fn = std::forward<F>(fn)](const NodeFactoryParams& p) -> std::unique_ptr<Node> {
		C config = p.engineConfig ? *static_cast<const C*>(p.engineConfig) : C{};
		return std::make_unique<Node>(
			engineType, p.nodeName, schema,
			[fn, config = std::move(config)](Node::RunContext& ctx) -> Node::Result { return fn(ctx, config); },
			ThreadPoolAffinity::Compute);
	};
}

// 带引擎实例版本：自动从 NodeFactoryParams 提取共享句柄并传给节点构造
// 同时注入 engineInstance->descriptor()，消除 EngineRegistry::instance() 隐式依赖
// 注意：句柄引用由节点持有，引擎实例存活期覆盖节点存活期
template <typename F>
NodeFactory makeNodeFactoryWithEngine(std::string engineType, Node::Schema schema, F&& fn) {
	return [engineType = std::move(engineType), schema = std::move(schema),
			fn = std::forward<F>(fn)](const NodeFactoryParams& p) -> std::unique_ptr<Node> {
		auto node = std::make_unique<Node>(engineType, p.nodeName, schema, fn,
									  ThreadPoolAffinity::Compute);
		if (p.engineInstance)
			node->bindEngine(p.engineInstance, p.engineInstance->descriptor());
		return node;
	};
}

} // namespace DC
