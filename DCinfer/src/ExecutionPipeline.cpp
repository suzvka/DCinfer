#include "Node/internal/ExecutionPipeline.h"
#include "Node/internal/NodeExecState.h"
#include "Node/internal/TaskBuffer.h"
#include "Node/internal/SlotWorkspace.h"
#include "Node/internal/EngineAdapter.h"
#include "Node.h"

namespace DC {

NodeResult ExecutionPipeline::execute(
	const TaskId& taskId,
	const Node& node,
	NodeExecState& exec,
	NodeExecutionGate& gate) {

	auto& buffer = exec.buffer;
	auto& workspace = *exec.workspace;
	auto& engine = node.engine();
	const auto& schema = node.schema();
	const auto& fn = node.runFn();
	const auto& onComplete = node.completionCallback();

	// ⓪ 就绪预检：必选输入未就绪则拒绝执行（原 Node::tryExecute 语义）
	if (!node.isReady(taskId, buffer)) {
		throw NodeException(NodeException::ErrorType::NotReady, "ExecutionPipeline::execute",
							"task '" + taskId + "' is not ready");
	}

	// ⓪½ 节点闸租约：同一节点同时只允许一个 task 执行（Reentrant 语义）
	if (!gate.tryAcquire()) {
		throw NodeException(NodeException::ErrorType::Reentrant, "ExecutionPipeline::execute",
							"node '" + node.name() + "' is busy executing another task");
	}
	gate.setCurrentTask(taskId);

	NodeResult result;

	try {
		// ① 加载输入：task 缓冲区 → 工作输入槽位
		buffer.drainInputsTo(taskId, workspace, schema);

		// ② 清空上一轮工作输出
		workspace.clearOutputs();

		// ②½ preRun 钩子：推理前准备
		engine.preRun();

		// ③ 执行 RunFn
		try {
			Node::RunContext ctx(workspace, engine, schema, node.type(), node.name());
			result = fn(ctx);
		} catch (const std::exception& e) {
			result.status = NodeStatus::ExecutionFailed;
			result.message = e.what();
		} catch (...) {
			result.status = NodeStatus::ExecutionFailed;
			result.message = "Unknown exception in RunFn";
		}

		// ③¼ onError 钩子：执行失败时重置引擎状态
		if (!result.ok()) {
			engine.onError();
		}

		// ③½ 同步：确保异步引擎计算已完成
		if (result.ok()) {
			engine.synchronize();
		}

		// ③¾ postRun 钩子：同步后的后处理
		if (result.ok()) {
			Node::RunContext ctx(workspace, engine, schema, node.type(), node.name());
			engine.postRun(ctx);
		}

		// ④ 保存输出：工作输出槽位 → task 输出缓冲区
		buffer.fillOutputsFrom(taskId, workspace, schema);

		// ⑤ 验证输出完整性
		if (result.ok() && !buffer.validateOutputs(taskId, schema)) {
			result.status = NodeStatus::InternalError;
			result.message = "Not all required outputs were produced by RunFn";
		}

		// ⑥ 清理输入缓冲（输出缓冲保留，供调用方拉取）
		buffer.eraseInputs(taskId);

		// ⑦ 调用回调
		if (onComplete) {
			onComplete(taskId, result);
		}
	} catch (const std::exception& e) {
		// 加载阶段或执行阶段抛出未捕获异常，必须通知完成回调
		result.status = NodeStatus::ExecutionFailed;
		result.message = e.what();
		if (onComplete) {
			onComplete(taskId, result);
		}
		gate.clearCurrentTask();
		gate.release();
		throw;
	}

	gate.clearCurrentTask();
	gate.release();
	return result;
}

} // namespace DC
