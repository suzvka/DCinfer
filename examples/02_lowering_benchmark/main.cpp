// 02_lowering_benchmark - Broadcast(1) lowering 内省与延迟基线
//
// 展示 Build → Freeze → Execute 边界的量化收益：
//   1. 构建链式图（N 个业务节点经 connect() 自动插入 1:1 导线连接器）
//   2. freeze() 后对比 源图视角 vs 运行时视图 的节点/边数量
//   3. 重复提交任务测量平均端到端延迟（lowering 减少调度顶点与传播跳数）
//
// 预期输出：runtimeNodeCount = 源图节点数 − 被擦除导线数

#include "InferGraph.h"
#include "GraphBuilder.h"

#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace DC;

namespace {

Node::Schema idSchema() {
	Node::Schema s;
	s.inputs = {{"x", Tensor::TensorType::Float, sizeof(float), {}}};
	s.outputs = {{"y", Tensor::TensorType::Float, sizeof(float), {}}};
	return s;
}

Node::RunFn idRunFn() {
	return [](Node::RunContext& ctx) -> Node::Result {
		ctx.output("y", ctx.pop("x"));
		return ctx.success();
	};
}

} // namespace

int main(int argc, char** argv) {
	const int chainLength = argc > 1 ? std::atoi(argv[1]) : 100; // 业务节点数
	const int taskCount = argc > 2 ? std::atoi(argv[2]) : 200;   // 提交任务数

	// ── 1. 构建链式图：id_0 → id_1 → ... → id_{N-1}（connect 自动插 wire）──
	InferGraph graph;
	for (int i = 0; i < chainLength; ++i) {
		graph.addNode(std::make_unique<Node>("Builtin", "id_" + std::to_string(i),
											 idSchema(), idRunFn()));
	}
	for (int i = 0; i + 1 < chainLength; ++i)
		graph.connect("id_" + std::to_string(i), "y", "id_" + std::to_string(i + 1), "x");

	graph.bindInput("in", "id_0", "x");
	graph.bindOutput("out", "id_" + std::to_string(chainLength - 1), "y");

	// ── 2. 冻结并对比源图视角 vs 运行时视图 ──
	auto snapshot = graph.freeze();

	std::cout << "chain length (business nodes): " << chainLength << "\n";
	std::cout << "source view : nodes=" << graph.nodeCount() << " edges=" << graph.edgeCount()
			  << "\n";
	std::cout << "runtime view: nodes=" << snapshot->runtimeNodeCount()
			  << " edges=" << snapshot->runtimeEdgeCount() << "\n";
	std::cout << "lowering    : erased connectors=" << snapshot->loweringStats().erasedConnectors
			  << "\n";

	// ── 3. 重复提交测量平均端到端延迟 ──
	using Clock = std::chrono::steady_clock;
	double totalMs = 0.0;
	float lastResult = 0.0f;
	for (int t = 0; t < taskCount; ++t) {
		const std::string taskId = "t_" + std::to_string(t);
		auto in = Tensor::Create<float>();
		in = static_cast<float>(t);
		auto start = Clock::now();
		graph.feedInput(taskId, "id_0", "x", std::move(in));                // 内部寻址注入
		graph.submitBound(taskId);
		graph.waitForResult(taskId);
		totalMs += std::chrono::duration<double, std::milli>(Clock::now() - start).count();
		lastResult =
			graph.takeOutputTensor(taskId, "id_" + std::to_string(chainLength - 1), "y").item<float>();
	}

	std::cout << "tasks=" << taskCount << " avg latency=" << totalMs / taskCount << " ms\n";
	std::cout << "last result=" << lastResult << " (expected " << (taskCount - 1) << ")\n";
	return std::fabs(lastResult - static_cast<float>(taskCount - 1)) < 1e-3f ? 0 : 1;
}
