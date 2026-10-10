#include "Node/internal/ExecutionPipeline.h"
#include "Node/internal/NodeExecState.h"
#include "Node/internal/TaskBuffer.h"
#include "Node/internal/SlotWorkspace.h"
#include "Node/internal/EngineAdapter.h"
#include "Node.h"

namespace DC {

namespace {

/// @brief 触发 onError 复位；onError 自身抛异常时吞掉
void safeTriggerOnError(EngineAdapter& engine) {
	try {
		engine.onError();
	} catch (...) {
	}
}

} // namespace

NodeResult ExecutionPipeline::execute(
	const TaskId& taskId,
	const Node& node,
	NodeExecState& exec,
	NodeExecutionGate& gate,
	std::function<bool()> isCancelRequested) {

	auto& buffer = exec.buffer;
	auto& workspace = *exec.workspace;
	auto& engine = node.engine();
	const auto& schema = node.schema();
	const auto& fn = node.runFn();
	const auto& onComplete = node.completionCallback();

	if (!node.isReady(taskId, buffer)) {
		throw NodeException(NodeException::ErrorType::NotReady, "ExecutionPipeline::execute",
							"task '" + taskId + "' is not ready");
	}

	// 节点闸租约：同节点同时只允许一个 task 执行
	if (!gate.tryAcquire()) {
		throw NodeException(NodeException::ErrorType::Reentrant, "ExecutionPipeline::execute",
							"node '" + node.name() + "' is busy executing another task");
	}

	// 租约 RAII：任何退出路径都必须释放；闸泄漏会让重试登记永久滞留。
	struct GateGuard {
		NodeExecutionGate& gate;
		~GateGuard() { gate.release(); }
	} gateGuard{gate};

	NodeResult result;

	// 完成回调至多一次：共用同一门闩，重复通知为 no-op，不覆盖首次错误语义。
	bool completed = false;
	auto notifyOnce = [&](const NodeResult& r) {
		if (!onComplete || completed)
			return;
		completed = true;
		onComplete(taskId, r);
	};

	try {
		buffer.drainInputsTo(taskId, workspace, schema);

		workspace.clearOutputs();

		// preRun 钩子：推理前引擎级准备，失败同样触发 onError 复位
		try {
			engine.preRun();
		} catch (...) {
			safeTriggerOnError(engine);
			throw;
		}

		try {
			Node::RunContext ctx(workspace, engine, schema, node.type(), node.name(),
								 taskId, isCancelRequested);
			result = fn(ctx);
		} catch (const std::exception& e) {
			result.status = NodeStatus::ExecutionFailed;
			result.message = e.what();
		} catch (...) {
			result.status = NodeStatus::ExecutionFailed;
			result.message = "Unknown exception in RunFn";
		}

		// onError 钩子：任一引擎相位失败时复位引擎状态
		if (!result.ok()) {
			safeTriggerOnError(engine);
		}

		// 同步：确保异步引擎计算已完成，仅成功路径；失败触发 onError
		if (result.ok()) {
			try {
				engine.synchronize();
			} catch (...) {
				safeTriggerOnError(engine);
				throw;
			}
		}

		// postRun 钩子：同步后处理，仅成功路径；失败触发 onError
		if (result.ok()) {
			try {
				Node::RunContext ctx(workspace, engine, schema, node.type(), node.name(),
									 taskId, isCancelRequested);
				engine.postRun(ctx);
			} catch (...) {
				safeTriggerOnError(engine);
				throw;
			}
		}

		buffer.fillOutputsFrom(taskId, workspace, schema);

		if (result.ok() && !buffer.validateOutputs(taskId, schema)) {
			result.status = NodeStatus::InternalError;
			result.message = "Not all required outputs were produced by RunFn";
		}

		// 清理输入缓冲；输出缓冲保留供调用方拉取
		buffer.eraseInputs(taskId);

		notifyOnce(result);
	} catch (const std::exception& e) {
		// 未捕获异常仍须通知完成回调，经门闩可能 no-op；租约由 GateGuard 释放
		result.status = NodeStatus::ExecutionFailed;
		result.message = e.what();
		notifyOnce(result);
		throw;
	}

	return result;
}

} // namespace DC
