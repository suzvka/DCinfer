#pragma once

#include <atomic>
#include <optional>
#include <string>

namespace DC {

/// @brief 节点级执行闸：同一节点同一时刻只允许一个 task 进入执行流水线。
///
/// 生命周期语义:
/// - 图路径:由 TaskExecutionDomain 在冻结期按节点集合预建(结构此后不变,
///   运行期只翻标志位),与 CompiledGraph 同生命周期;
/// - 单节点路径(NetServerAdapter / 测试):由 NodeExecutor 持有,随载体走。
/// currentTask 的写入仅发生在持有租约期间。
class NodeExecutionGate {
public:
	NodeExecutionGate() = default;

	NodeExecutionGate(const NodeExecutionGate&) = delete;
	NodeExecutionGate& operator=(const NodeExecutionGate&) = delete;

	/// @brief  尝试获取执行租约(test_and_set)
	/// @return false = 已有 task 在执行(重入拒绝)
	bool tryAcquire() noexcept {
		return !_guard.test_and_set();
	}

	/// @brief  释放执行租约(须与成功的 tryAcquire 配对)
	void release() noexcept {
		_guard.clear();
	}

	/// @brief  记录当前执行中的 task ID(调用方须持有租约)
	void setCurrentTask(std::string taskId) {
		_currentTaskId = std::move(taskId);
	}

	/// @brief  清除当前 task ID(调用方须持有租约)
	void clearCurrentTask() noexcept {
		_currentTaskId.reset();
	}

	/// @brief  当前执行中的 task ID(诊断用;可能读到正在变更的值)
	std::optional<std::string> currentTask() const {
		return _currentTaskId;
	}

private:
	std::atomic_flag _guard = ATOMIC_FLAG_INIT;
	std::optional<std::string> _currentTaskId;
};

} // namespace DC
