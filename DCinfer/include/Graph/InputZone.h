#pragma once

#include <mutex>
#include <string>
#include <vector>

namespace DC {

struct InputBinding {
	std::string nodeName;
	std::string portName;
	std::string alias;
};

/// @brief 图级输入端口声明区，纯结构无 task 级状态；公开方法线程安全。
class InputZone {
public:
	/// @brief 标记 node:port 为图级输入口；alias 必填，构成图级签名。
	void bind(const std::string& nodeName, const std::string& portName,
			  const std::string& alias);

	/// @brief 获取所有输入绑定的值副本，按插入顺序。
	std::vector<InputBinding> bindings() const;

private:
	mutable std::mutex _mutex;
	std::vector<InputBinding> _bindingsList;
};

inline void InputZone::bind(const std::string& nodeName,
							const std::string& portName,
							const std::string& alias) {
	std::lock_guard lk(_mutex);
	_bindingsList.push_back({nodeName, portName, alias});
}

inline std::vector<InputBinding> InputZone::bindings() const {
	std::lock_guard lk(_mutex);
	return _bindingsList;
}

} // namespace DC
