// 04_task_lifecycle - 统一任务句柄：同步与异步同级
//
// 同一套 GraphInterface::Task 的两种节奏（同一 Add → Identity 图）：
//   1. 同步一发：feed 链式注入 → run()（内部 submit + 无限等待）
//   2. 异步：feed + submit 后做其他工作，再 wait(超时) + take；
//      终态任务随句柄析构自动释放；在飞任务不受析构影响（不取消）
//
// 全部按公开别名操作，无 taskId / 坐标。宿主唯一 API 见 examples/01。
//
// 预期输出：3.0 + 4.0 = 7 / 10.0 + 5.0 = 15

#include "InferGraph.h"
#include "Tensor.hpp"
#include "DCEngine/BuiltinOps.h"

#include <chrono>
#include <iostream>

int main() {
	// ── 1. 注册内置 CPU 算子 ──
	DC::Builtin::registerBuiltinOperators();

	// ── 2. 建图：Add → Identity，绑定图级输入输出 ──
	auto& reg = DC::EngineRegistry::instance();
	DC::InferGraph graph;
	graph.addNode(reg.createOperator("Add", "adder"));
	graph.addNode(reg.createOperator("Identity", "pass"));
	graph.connect("adder", "sum", "pass", "x");
	graph.bindInput("a", "adder", "a");
	graph.bindInput("b", "adder", "b");
	graph.bindOutput("result", "pass", "y");

	auto api = graph.interface(); // 取接口即定型：冻结图并解析别名 → 坐标

	// ── 3. 同步一发：run() = submit + wait（同一句柄）──
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

	// ── 4. 异步：同一句柄换一种节奏——submit 后做其他工作，再 wait(超时) + take ──
	float asyncValue = 0.0f;
	{
		auto task = api.createTask();
		auto ta = DC::Tensor::Create<float>();
		ta = 10.0f;
		auto tb = DC::Tensor::Create<float>();
		tb = 5.0f;
		task.feed("a", std::move(ta)).feed("b", std::move(tb)).submit(); // 异步启动（不等待）

		// …… 此处可执行与本次推理无关的其他工作 ……（同步/异步同级的差别仅在此处）

		auto result = task.wait(std::chrono::milliseconds(5000));
		if (result.status == DC::TaskStatus::Succeeded)
			asyncValue = task.takeTensor("result").item<float>();
	} // 在飞任务不受析构影响；显式放弃可调用 task.cancel()
	std::cout << "10.0 + 5.0 = " << asyncValue << std::endl;

	return 0;
}