// 01_hello_graph - 最简推理图示例
//
// 展示完整的 DCinfer 使用流程：
//   1. 注册算子
//   2. 构建图（Add → Identity）
//   3. 注入数据
//   4. 提交并等待
//   5. 获取结果
//
// 预期输出：3 + 4 = 7

#include "InferGraph.h"
#include "Tensor.hpp"
#include "DCEngine/BuiltinOps.h"

#include <iostream>
#include <optional>

int main() {
	// ── 1. 注册内置 CPU 算子 ──
	DC::Builtin::registerBuiltinOperators();

	// ── 2. 通过 Registry 创建节点 ──
	auto& reg = DC::EngineRegistry::instance();
	auto addNode = reg.createOperator("Add", "adder");    // 输入 a, b → 输出 sum
	auto idNode  = reg.createOperator("Identity", "pass"); // 输入 x → 输出 y

	// ── 3. 构建推理图 ──
	DC::InferGraph graph;
	graph.addNode(std::move(addNode));
	graph.addNode(std::move(idNode));

	// 连接 adder.sum → pass.x（wire 自动插入广播连接器）
	graph.wire("adder", "sum", "pass", "x");

	// 标记图级输入输出端口
	graph.bindInput("adder", "a");
	graph.bindInput("adder", "b");
	graph.bindOutput("pass", "y");

	// ── 4. 注入数据 ──
	auto tensorA = DC::Tensor::Create<float>();
	tensorA = 3.0f;
	auto tensorB = DC::Tensor::Create<float>();
	tensorB = 4.0f;

	graph.feedInput("task1", "adder", "a", std::move(tensorA));
	graph.feedInput("task1", "adder", "b", std::move(tensorB));

	// ── 5. 声明输出并异步提交 ──
	// 注意：task 完成时输出缓冲区会被清理，必须在完成回调中捕获结果
	std::optional<DC::Tensor> captured;
	graph.setTaskCompleteCallback([&graph, &captured](const DC::InferGraph::TaskId& taskId) {
		if (graph.hasOutput(taskId, "pass", "y")) {
			auto val = graph.getOutput(taskId, "pass", "y");
			if (auto* t = val.as<DC::Tensor>())
				captured = std::move(*t);
		}
	});
	graph.submit("task1", "pass", "y");

	// ── 6. 等待完成 ──
	if (!graph.wait("task1")) {
		std::cerr << "Error: task timed out" << std::endl;
		for (const auto& err : graph.taskErrors("task1"))
			std::cerr << "  " << err.nodeName << ": " << err.message << std::endl;
		return 1;
	}
	if (!captured) {
		std::cerr << "Error: no output captured" << std::endl;
		for (const auto& err : graph.taskErrors("task1"))
			std::cerr << "  " << err.nodeName << ": " << err.message << std::endl;
		return 1;
	}

	// ── 7. 获取结果 ──
	std::cout << "3.0 + 4.0 = " << captured->item<float>() << std::endl;

	return 0;
}
