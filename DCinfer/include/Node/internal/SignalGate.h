#pragma once

#include <memory>
#include <string>

namespace DC {

class SignalStore;

/// @brief 信号阻塞门：节点信号判断逻辑的独立封装；未绑定时 isBlocked 恒 false。
class SignalGate {
public:
	SignalGate() = default;

	/// @brief 绑定信号存储与信号名。
	void bind(std::shared_ptr<SignalStore> store, std::string name);

	/// @brief 是否被信号阻塞；只查全局信号，false 值表示阻塞。
	bool isBlocked() const;

	/// @brief 是否被信号阻塞；task 级优先，全局回退，最后 false。
	bool isBlocked(const std::string& taskId) const;

private:
	std::shared_ptr<SignalStore> _store;
	std::string _name;
};

} // namespace DC
