#pragma once

#include <memory>
#include <string>

namespace DC {

class SignalStore;

/// @brief 信号阻塞门：将信号判断逻辑从 Node 中解耦。
///
/// 与数据图正交解耦，任意时刻可读写。
/// 未绑定时 isBlocked() 永远返回 false（向后兼容）。
class SignalGate {
public:
	SignalGate() = default;

	/// @brief  绑定图级信号到此节点。
	/// @param store 信号仓库指针（共享所有权）。
	/// @param name  信号名。
	void bind(std::shared_ptr<SignalStore> store, std::string name);

	/// @brief  查询节点是否被信号阻塞（只查全局信号）。
	///         signal==false → 阻塞（true）；signal==true 或未绑定 → 不阻塞（false）。
	bool isBlocked() const;

	/// @brief  查询节点是否被信号阻塞（带 taskId，task 级信号优先）。
	///         查找顺序：task 级信号 → 全局信号 → 默认值(false)。
	bool isBlocked(const std::string& taskId) const;

private:
	std::shared_ptr<SignalStore> _store;
	std::string _name;
};

} // namespace DC
