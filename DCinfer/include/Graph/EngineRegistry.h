#pragma once

#include "Node.h"
#include "Value.h"

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace DC {

// ── 类型擦除的模型句柄 ──
// 封装引擎内部模型对象（Ort::Session* / nvinfer1::ICudaEngine* / …）
// 基于 shared_ptr<void>，自动保留原始 deleter，引擎内部 static_cast 还原
class ModelHandle {
public:
	ModelHandle() = default;

	template <typename T>
	explicit ModelHandle(std::shared_ptr<T> ptr) : _ptr(std::move(ptr)) {}

	void* get() {
		return _ptr.get();
	}
	const void* get() const {
		return _ptr.get();
	}

	explicit operator bool() const {
		return _ptr != nullptr;
	}

private:
	std::shared_ptr<void> _ptr;
};

// ── 引擎实例：类型擦除的运行时引擎句柄 ──
// 封装引擎运行时对象（Ort::Session / nvinfer1::ICudaEngine / 自定义对象）
// 基于 shared_ptr<void> 实现类型擦除，EngineDescriptor 充当虚表
class EngineInstance {
public:
	EngineInstance() = default;

	template <typename T>
	EngineInstance(std::shared_ptr<T> engine, const EngineDescriptor* desc) : _engine(std::move(engine)), _desc(desc) {}

	void* get() {
		return _engine.get();
	}
	const void* get() const {
		return _engine.get();
	}

	const EngineDescriptor* descriptor() const {
		return _desc;
	}

	explicit operator bool() const {
		return _engine != nullptr;
	}

private:
	std::shared_ptr<void> _engine;
	const EngineDescriptor* _desc = nullptr;
};

// ── 引擎描述符：注册一个引擎所需的全部信息 ──
struct EngineDescriptor {
	std::string engineType;
	TensorConverter converter;
	NodeFactory factory;

	// ── 可选：模型级钩子 ──

	/// 从路径加载模型，返回类型擦除的模型句柄
	/// 对于 Builtin 引擎，此钩子为 nullptr
	std::function<ModelHandle(const std::string& path)> loadModel;

	/// 从已加载的模型句柄获取输入端口列表
	/// 用于自动推导 Schema，消除用户手动声明
	std::function<std::vector<Node::Port>(ModelHandle)> getInputPorts;

	/// 从已加载的模型句柄获取输出端口列表
	std::function<std::vector<Node::Port>(ModelHandle)> getOutputPorts;

	// ── 可选：运行时钩子 ──

	/// 从模型路径创建引擎实例（含运行时资源分配）
	/// 系统以 modelPath 为 key 缓存实例，用户可在钩子内自行决定复用策略
	std::function<EngineInstance(const std::string& modelPath)> createEngine;

	/// RunFn 返回后、输出收集前调用，确保异步计算已完成
	/// engine 指针为 EngineInstance::get() 返回的原生指针
	std::function<void(void* engine)> synchronize;

	// ── 运行时优化钩子 ──

	/// 每次 RunFn 调用前执行，用于推理前准备（I/O 绑定、warmup、动态 shape 设置等）
	/// engine 为 EngineInstance::get() 返回的原生指针
	std::function<void(void* engine)> preRun;

	/// synchronize 成功后、输出收集前调用
	/// 典型用途：device→host 数据传输、输出格式后处理
	/// ctx 提供完整的输入/输出槽位访问，可通过 ctx.outputRaw() 读取 GPU 输出、
	/// ctx.output() 写回 host 数据
	std::function<void(void* engine, Node::RunContext& ctx)> postRun;

	/// 引擎实例释放前调用，用于有序清理 GPU 资源
	/// 若为 nullptr，退化为 shared_ptr<void> 默认析构
	std::function<void(void* engine)> releaseEngine;

	/// RunFn 失败或抛出异常后调用，用于引擎状态重置
	std::function<void(void* engine)> onError;
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
	// 自动调用 getOrCreateEngine → getInputPorts → getOutputPorts → 构建 Schema
	// 节点持有 EngineInstance 的非拥有引用，引擎生命周期由 Registry 管理
	std::unique_ptr<Node> createNode(const std::string& engineType, const std::string& nodeName,
									 const std::string& modelPath);

	// ── 引擎实例管理 ──

	/// 获取或创建引擎实例（以 engineType + modelPath 复合键缓存）
	/// 若未缓存则调用 EngineDescriptor::createEngine 创建
	/// 返回非拥有指针，由 Registry 统一管理生命周期
	EngineInstance* getOrCreateEngine(const std::string& engineType, const std::string& modelPath);

	/// 释放指定引擎类型 + 模型路径的引擎实例
	void releaseEngine(const std::string& engineType, const std::string& modelPath);

	/// 释放所有引擎实例
	void releaseAllEngines();

	const EngineDescriptor* find(const std::string& engineType) const;
	bool hasEngine(const std::string& engineType) const;
	std::vector<std::string> engineTypes() const;

	// ── 算子注册（轻量级，DC::Tensor only，无引擎钩子）──

	/// @brief  注册一个算子节点类型
	/// @param  operatorName  算子名（如 "Broadcast", "Routing", "Add"）
	/// @param  schema        输入/输出端口 Schema
	/// @param  fn            算子计算逻辑
	/// @return true 表示注册成功，false 表示已存在同名算子
	bool registerOperator(const std::string& operatorName, Node::Schema schema, Node::RunFn fn);

	/// @brief  从已注册算子创建节点（无需 engineConfig）
	std::unique_ptr<Node> createOperator(const std::string& operatorName, const std::string& nodeName) const;

private:
	EngineRegistry() = default;

	static std::string _makeEngineKey(const std::string& engineType, const std::string& modelPath);

	std::unordered_map<std::string, EngineDescriptor> _engines;
	std::unordered_map<std::string, EngineInstance> _engineInstances;
};

// ── 节点工厂辅助模板 ──
// 无配置版本
template <typename F>
NodeFactory makeNodeFactory(std::string engineType, Node::Schema schema, F&& fn) {
	return [engineType = std::move(engineType), schema = std::move(schema),
			fn = std::forward<F>(fn)](std::string name, const void* /*engineConfig*/) -> std::unique_ptr<Node> {
		return std::make_unique<Node>(engineType, std::move(name), schema, fn, nullptr, ThreadPoolAffinity::Compute);
	};
}

// 带配置版本：engineConfig 在 createNode 时拷贝为值，lambda 接收 const C&
template <typename C, typename F>
NodeFactory makeNodeFactory(std::string engineType, Node::Schema schema, F&& fn) {
	return [engineType = std::move(engineType), schema = std::move(schema),
			fn = std::forward<F>(fn)](std::string name, const void* engineConfig) -> std::unique_ptr<Node> {
		C config = engineConfig ? *static_cast<const C*>(engineConfig) : C{};
		return std::make_unique<Node>(
			engineType, std::move(name), schema,
			[fn, config = std::move(config)](Node::RunContext& ctx) -> Node::Result { return fn(ctx, config); },
			nullptr, ThreadPoolAffinity::Compute);
	};
}

// 带引擎实例版本：自动从 engineConfig 提取 EngineInstance* 并传给节点构造
// 同时注入 engineInstance->descriptor()，消除 EngineRegistry::instance() 隐式依赖
template <typename F>
NodeFactory makeNodeFactoryWithEngine(std::string engineType, Node::Schema schema, F&& fn) {
	return [engineType = std::move(engineType), schema = std::move(schema),
			fn = std::forward<F>(fn)](std::string name, const void* engineConfig) -> std::unique_ptr<Node> {
		auto* engineInstance = engineConfig ? *static_cast<EngineInstance* const*>(engineConfig) : nullptr;
		return std::make_unique<Node>(engineType, std::move(name), schema, fn, engineInstance,
									  ThreadPoolAffinity::Compute,
									  engineInstance ? engineInstance->descriptor() : nullptr);
	};
}

} // namespace DC
