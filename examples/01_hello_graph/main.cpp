// 01_hello_graph - 最简推理图示例：注册算子 → 构建图（Add → Identity）→ 取公开接口 →
// 按别名注入数据、提交并取结果。预期输出：3 + 4 = 7

#include "InferGraph.h"
#include "Tensor.hpp"
#include "DCEngine/BuiltinOps.h"

#include <iostream>

int main() {
	DC::Builtin::registerBuiltinOperators();

	auto& reg = DC::EngineRegistry::instance();
	auto addNode = reg.createOperator("Add", "adder");
	auto idNode  = reg.createOperator("Identity", "pass");

	DC::InferGraph graph;
	graph.addNode(std::move(addNode));
	graph.addNode(std::move(idNode));

	// connect 自动插入广播连接器
	graph.connect("adder", "sum", "pass", "x");

	// 声明图公开接口：别名 → 内部端口（构成图级签名）
	graph.bindInput("a", "adder", "a");
	graph.bindInput("b", "adder", "b");
	graph.bindOutput("result", "pass", "y");

	// 取公开接口：冻结图并一次性解析别名 → 坐标
	auto api = graph.interface();
	auto task = api.createTask(); // 任务句柄：析构自动释放已终止任务

	auto tensorA = DC::Tensor::Create<float>();
	tensorA = 3.0f;
	auto tensorB = DC::Tensor::Create<float>();
	tensorB = 4.0f;

	task.feed("a", std::move(tensorA)).feed("b", std::move(tensorB));

	// 同步运行（内部 submitBound + 等待终止）
	auto result = task.run();
	if (result.status != DC::TaskStatus::Succeeded) {
		std::cerr << "Error: task ended with status " << static_cast<int>(result.status)
				  << std::endl;
		for (const auto& err : result.errors)
			std::cerr << "  " << err.nodeName << ": " << err.message << std::endl;
		return 1;
	}

	// 取出即消耗
	auto output = task.takeTensor("result");
	std::cout << "3.0 + 4.0 = " << output.item<float>() << std::endl;

	return 0;
}
