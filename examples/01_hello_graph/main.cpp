// 01_hello_graph - 最简推理图示例
//
// 展示完整的 DCinfer 使用流程：
//   1. 注册算子
//   2. 构建图（Add → Identity）
//   3. 取公开接口
//   4. 按别名注入数据并提交
//   5. 按别名获取结果
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

	// 连接 adder.sum → pass.x（connect 自动插入广播连接器）
	graph.connect("adder", "sum", "pass", "x");

	// 声明图公开接口：别名 → 内部端口（绑定构成图级签名：submitBound 声明来源 / 序列化与内省元数据）
	graph.bindInput("a", "adder", "a");
	graph.bindInput("b", "adder", "b");
	graph.bindOutput("result", "pass", "y");   // 图级输出别名 result → 内部 pass.y

	// ── 4. 取公开接口：冻结图并一次性解析别名 → 坐标 ──
	auto api = graph.interface();
	auto task = api.createTask(); // 任务句柄：析构自动释放已终止任务

	// ── 5. 按公开别名注入数据 ──
	auto tensorA = DC::Tensor::Create<float>();
	tensorA = 3.0f;
	auto tensorB = DC::Tensor::Create<float>();
	tensorB = 4.0f;

	task.feed("a", std::move(tensorA));
	task.feed("b", std::move(tensorB));

	// ── 6. 同步运行并获取结构化结果（内部 submitBound + 等待终止）──
	auto result = task.run();
	if (result.status != DC::TaskStatus::Succeeded) {
		std::cerr << "Error: task ended with status " << static_cast<int>(result.status)
				  << std::endl;
		for (const auto& err : result.errors)
			std::cerr << "  " << err.nodeName << ": " << err.message << std::endl;
		return 1;
	}

	// ── 7. 按公开别名取结果（取出即消耗）──
	auto output = task.takeTensor("result");
	std::cout << "3.0 + 4.0 = " << output.item<float>() << std::endl;

	return 0;
}
