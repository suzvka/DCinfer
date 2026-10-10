// 03_custom_node - 编写自定义节点的完整教程。
//
// 从零编写可注册算子（Scale：y = x * factor）的四步流程：
//   1. NodePort 工厂声明端口 Schema；2. RunFn 用 ctx.input<Tensor>() 类型化读取、
//   经 ctx.failure() 报结构化错误；3. registerOperator 注册；4. createOperator 建图执行。
//
// 端口 Schema 两种等价写法（推荐工厂：类型与 typeSize 单点书写）：
//   s.inputs = {Node::Port::in<float>("x")};                            // 工厂（推荐）
//   s.inputs = {{"x", Tensor::TensorType::Float, sizeof(float), {}}};   // 聚合初始化
//
// 可选默认值：Node::Port::optional<float>("name", 1.0f)；形状锚定：
// Node::Port::anchored<float>("name", "anchorPort")。预期输出：3.0 * 2.5 = 7.5

#include "InferGraph.h"
#include "EngineRegistry.h"
#include "Tensor.hpp"

#include <cmath>
#include <iostream>
#include <memory>

using namespace DC;

namespace {

// Schema：端口用工厂声明（shape 省略 = 标量）
Node::Schema scaleSchema() {
	Node::Schema s;
	s.inputs = {Node::Port::in<float>("x")};
	s.outputs = {Node::Port::out<float>("y")};
	return s;
}

// factor 为节点私有参数（闭包捕获）；真实算子可改为 engineConfig 或输入端口。
Node::RunFn scaleRunFn(float factor) {
	return [factor](Node::RunContext& ctx) -> Node::Result {
		const auto* x = ctx.input<Tensor>("x"); // peek + 类型校验 + 空值检查的组合
		if (!x)
			return ctx.failure(Node::Status::InvalidInput, "scale: 'x' must be a float Tensor");

		auto out = std::make_unique<Tensor>(Tensor::TensorType::Float, sizeof(float));
		*out = x->item<float>() * factor;
		ctx.output("y", Value(std::move(out)));
		return ctx.success();
	};
}

} // namespace

int main() {
	// 注册：轻量算子（DC::Tensor only、无引擎钩子）
	auto& reg = EngineRegistry::instance();
	if (!reg.registerOperator("Scale", scaleSchema(), scaleRunFn(2.5f))) {
		std::cerr << "register failed: operator 'Scale' already exists" << std::endl;
		return 1;
	}

	// 建图：createOperator → addNode → 绑定图级输入输出
	InferGraph graph;
	graph.addNode(reg.createOperator("Scale", "scale1"));
	graph.bindInput("in", "scale1", "x");
	graph.bindOutput("out", "scale1", "y");

	auto api = graph.interface(); // 冻结图并一次性解析 bindInput/bindOutput 别名
	auto task = api.createTask(); // 任务句柄：析构自动释放已终止任务

	auto in = Tensor::Create<float>();
	in = 3.0f;
	task.feed("in", std::move(in));

	auto result = task.run(); // 同步：提交全部绑定输出并等待终止
	if (result.status != TaskStatus::Succeeded) {
		std::cerr << "task failed" << std::endl;
		for (const auto& err : result.errors)
			std::cerr << "  " << err.nodeName << ": " << err.message << std::endl;
		return 1;
	}

	auto out = task.takeTensor("out");
	const float value = out.item<float>();
	std::cout << "3.0 * 2.5 = " << value << std::endl;

	return std::fabs(value - 7.5f) < 1e-6f ? 0 : 1;
}
