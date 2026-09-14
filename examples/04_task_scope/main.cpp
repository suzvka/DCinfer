// 04_task_scope - 内核坐标层的作用域句柄（TaskScope）
//
// 展示两种用法（同一 Add → Identity 图）：
//   1. 同步一发：feed 链式注入 → run()（submitBound → 等待 → 收割全部绑定输出
//      → 释放）；run 返回后任务已释放，析构为空操作
//   2. 异步作用域：feed + submit 后做其他工作，再 wait(超时) + take；
//      未取完 / 运行中退出作用域 → 析构取消并释放（销毁即放弃）
//
// 注：按公开别名操作的别名层便捷路径见 examples/01_hello_graph
//     （graph.interface() → createTask()）。
//
// 预期输出：3.0 + 4.0 = 7.0 / 10.0 + 5.0 = 15.0

#include "InferGraph.h"
#include "TaskScope.h"
#include "Tensor.hpp"
#include "DCEngine/BuiltinOps.h"

#include <chrono>
#include <iostream>

int main() {
	// ── 1. 注册内置 CPU 算子 ──
	DC::Builtin::registerBuiltinOperators();

	// ── 2. 建图：Add → Identity（内核坐标寻址；坐标即 (nodeName, portName)）──
	auto& reg = DC::EngineRegistry::instance();
	DC::InferGraph graph;
	graph.addNode(reg.createOperator("Add", "adder"));
	graph.addNode(reg.createOperator("Identity", "pass"));
	graph.connect("adder", "sum", "pass", "x");
	graph.bindInput("a", "adder", "a");
	graph.bindInput("b", "adder", "b");
	graph.bindOutput("result", "pass", "y"); // submitBound / run() 的声明来源

	// ── 3. 同步一发：三行完成一次推理 ──
	{
		DC::TaskScope task{graph, "task1"}; // 绑定 taskId（此刻尚无任务）
		auto ta = DC::Tensor::Create<float>();
		ta = 3.0f;
		auto tb = DC::Tensor::Create<float>();
		tb = 4.0f;
		task.feed("adder", "a", std::move(ta)).feed("adder", "b", std::move(tb));
		auto r = task.run();
		if (r.status != DC::TaskStatus::Succeeded) {
			std::cerr << "Error: task ended with status " << static_cast<int>(r.status)
					  << std::endl;
			for (const auto& err : r.errors)
				std::cerr << "  " << err.nodeName << ": " << err.message << std::endl;
			return 1;
		}
		// run() 已收割全部绑定输出并释放任务
		std::cout << "3.0 + 4.0 = " << r.takeTensor("pass", "y").item<float>() << std::endl;
	} // 任务已由 run() 释放：析构为空操作

	// ── 4. 异步作用域：submit 后做其他工作，再取结果 ──
	float asyncValue = 0.0f;
	{
		DC::TaskScope task{graph, "task2"};
		auto ta = DC::Tensor::Create<float>();
		ta = 10.0f;
		auto tb = DC::Tensor::Create<float>();
		tb = 5.0f;
		task.feed("adder", "a", std::move(ta)).feed("adder", "b", std::move(tb)).submit();

		// …… 此处可执行与本次推理无关的其他工作 ……

		auto result = task.wait(std::chrono::milliseconds(5000));
		if (result.status == DC::TaskStatus::Succeeded)
			asyncValue = task.takeTensor("pass", "y").item<float>();
	} // 若任务仍在运行，析构先 cancel（同步终止）再释放——销毁即放弃
	std::cout << "10.0 + 5.0 = " << asyncValue << std::endl;

	return 0;
}
