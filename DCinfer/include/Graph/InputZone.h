#pragma once

#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>

namespace DC {

/// @brief 输入绑定（序列化用）
struct InputBinding {
	std::string nodeName;
	std::string portName;
};

/// @brief InputZone：图级输入端口声明区（纯结构，无 task 级状态）
///
/// 语义：
/// - bind() 标记 node:port 为图级输入口，外部通过此口注入数据
/// - 与 OutputZone 完全对称，构成图的完整外部签名
/// - 不存储数据，数据注入仍通过 feedInput 透传给节点
///
/// 所有公开方法线程安全（内部 mutex）。
class InputZone {
public:
	/// @brief  标记 node:port 为图级输入口
	void bind(const std::string& nodeName, const std::string& portName);

	/// @brief  检查是否已绑定为图级输入
	bool isBound(const std::string& nodeName, const std::string& portName) const;

	/// @brief  获取所有输入绑定（只读，保留插入顺序）
	const std::vector<InputBinding>& bindings() const;

private:
	static std::string _makeKey(const std::string& nodeName,
								const std::string& portName) {
		return nodeName + ":" + portName;
	}

	mutable std::mutex _mutex;
	std::unordered_set<std::string> _bindings;   // "nodeName:portName" 去重
	std::vector<InputBinding> _bindingsList;      // 保持插入顺序，序列化用
};

// ════════════════════════════════════════════
// 内联实现
// ════════════════════════════════════════════

inline void InputZone::bind(const std::string& nodeName,
							const std::string& portName) {
	std::lock_guard lk(_mutex);
	std::string key = _makeKey(nodeName, portName);
	_bindings.insert(key);
	_bindingsList.push_back({nodeName, portName});
}

inline bool InputZone::isBound(const std::string& nodeName,
							   const std::string& portName) const {
	std::lock_guard lk(_mutex);
	return _bindings.contains(_makeKey(nodeName, portName));
}

inline const std::vector<InputBinding>& InputZone::bindings() const {
	std::lock_guard lk(_mutex);
	return _bindingsList;
}

} // namespace DC
