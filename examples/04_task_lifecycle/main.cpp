// 04_task_lifecycle：统一任务句柄，同步与异步同级
// 预期输出：3.0 + 4.0 = 7 与 10.0 + 5.0 = 15

#include "InferGraph.h"
#include "Tensor.hpp"
#include "DCEngine/BuiltinOps.h"

#include <chrono>
#include <iostream>

int main() {
	DC::Builtin::registerBuiltinOperators();

	auto& reg = DC::EngineRegistry::instance();
	DC::InferGraph graph;
	graph.addNode(reg.createOperator("Add", "adder"));
	graph.addNode(reg.createOperator("Identity", "pass"));
	graph.connect("adder", "sum", "pass", "x");
	graph.bindInput("a", "adder", "a");
	graph.bindInput("b", "adder", "b");
	graph.bindOutput("result", "pass", "y");

	auto api = graph.interface(); // 取接口即定型：冻结图并解析别名到坐标

	// 同步：run 即 submit 加 wait，同一句柄
	{
		auto task = api.createTask();
		auto ta = DC::Tensor::Create<float>();
		ta = 3.0f;
		auto tb = DC::Tensor::Create<float>();
		tb = 4.0f;
		auto r = task.feed("a", std::move(ta)).feed("b", std::move(tb)).run();
		if (r.status != DC::TaskStatus::Succeeded) {
			std::cerr << "Error: task ended with status " << static_cast<int>(r.status)
					  << std::endl;
			for (const auto& err : r.errors)
				std::cerr << "  " << err.nodeName << ": " << err.message << std::endl;
			return 1;
		}
		std::cout << "3.0 + 4.0 = " << task.takeTensor("result").item<float>() << std::endl;
	} // 终态任务随析构自动释放

	// 异步：submit 后做其他工作，再 wait 超时后 take
	float asyncValue = 0.0f;
	{
		auto task = api.createTask();
		auto ta = DC::Tensor::Create<float>();
		ta = 10.0f;
		auto tb = DC::Tensor::Create<float>();
		tb = 5.0f;
		task.feed("a", std::move(ta)).feed("b", std::move(tb)).submit();

		// …… 与本次推理无关的其他工作，同步与异步差别仅在此处 ……

		auto result = task.wait(std::chrono::milliseconds(5000));
		if (result.status == DC::TaskStatus::Succeeded)
			asyncValue = task.takeTensor("result").item<float>();
	} // 终态任务随析构释放；在飞弃置触发协作式取消
	std::cout << "10.0 + 5.0 = " << asyncValue << std::endl;

	return 0;
}