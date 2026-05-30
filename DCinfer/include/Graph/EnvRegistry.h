#pragma once

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

namespace DC {

// ── 环境注册表：管理引擎运行时环境的创建、缓存与生命周期 ──
//
// 每种引擎类型（如 "ONNX", "TensorRT"）可能需要一个全局共享的运行时环境
// （如 Ort::Env、CUDA context）。EnvRegistry 负责托管这些环境对象，
// 用户只需注册工厂函数，后续通过 getOrCreate() 按需获取。
//
// 类型擦除基于 shared_ptr<void>，自动保留原始 deleter。
// 遵循单例模式，与 EngineRegistry / ValidatorRegistry 一致。
//
// 用法：
//   EnvRegistry::instance().registerEnv("ONNX", []() {
//       return std::make_shared<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "dc");
//   });
//   auto* env = static_cast<Ort::Env*>(EnvRegistry::instance().getOrCreate("ONNX"));
//
// 释放顺序：先 releaseAllEngines()，再 releaseAll()。
class EnvRegistry {
public:
	static EnvRegistry& instance();

	/// @brief  注册一个环境工厂
	/// @param  envType   环境类型名（通常与引擎类型同名，如 "ONNX", "TensorRT"）
	/// @param  factory   创建环境的工厂函数，返回 shared_ptr<void>
	/// @param  cleanup   可选：释放前调用的清理钩子
	/// @return true 表示注册成功，false 表示已存在同名环境
	bool registerEnv(const std::string& envType,
					 std::function<std::shared_ptr<void>()> factory,
					 std::function<void(void*)> cleanup = nullptr);

	/// @brief  获取或创建环境实例（按 envType 缓存）
	/// @return 环境裸指针，若未注册则返回 nullptr
	void* getOrCreate(const std::string& envType);

	/// @brief  释放指定环境（若存在则先调用 cleanup 钩子再移除实例）
	void release(const std::string& envType);

	/// @brief  释放所有环境（依序调用所有 cleanup）
	void releaseAll();

	/// @brief  查询指定环境类型是否已注册
	bool hasEnv(const std::string& envType) const;

private:
	EnvRegistry() = default;

	struct Entry {
		std::function<std::shared_ptr<void>()> factory;
		std::function<void(void*)> cleanup;
	};

	std::unordered_map<std::string, Entry> _factories;
	std::unordered_map<std::string, std::shared_ptr<void>> _instances;
};

} // namespace DC
