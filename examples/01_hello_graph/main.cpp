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
	// 注：connect() 禁止两个业务节点直连（数据必须经 Connector 中转），
	// wire() 则自动插入广播连接器——这是两者唯一的语义差异。
	graph.bindInput("adder", "a");
	graph.bindInput("adder", "b");
	graph.bindOutput("pass", "y");

	// ── 4. 注入数据（按绑定端口名，无需重复提供节点名）──
	auto tensorA = DC::Tensor::Create<float>();
	tensorA = 3.0f;
	auto tensorB = DC::Tensor::Create<float>();
	tensorB = 4.0f;

	graph.feedBoundInput("task1", "a", std::move(tensorA));
	graph.feedBoundInput("task1", "b", std::move(tensorB));

	// ── 5. 提交（以全部 bindOutput 绑定作为输出声明，无需重复声明）──
	// 结果在任务终止后仍保留，无需注册回调即可在 wait 之后读取。
	graph.submitBound("task1");

	// ── 6. 等待完成并获取结构化结果 ──
	auto result = graph.waitForResult("task1");
	if (result.status != DC::TaskStatus::Succeeded) {
		std::cerr << "Error: task ended with status " << static_cast<int>(result.status)
				  << std::endl;
		for (const auto& err : result.errors)
			std::cerr << "  " << err.nodeName << ": " << err.message << std::endl;
		return 1;
	}

	// ── 7. 获取结果（wait 返回后仍然有效）──
	auto output = graph.getOutputTensor("task1", "pass", "y");
	std::cout << "3.0 + 4.0 = " << output.item<float>() << std::endl;

	return 0;
}
