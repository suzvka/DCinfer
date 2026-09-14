// 03_custom_node - 编写自定义节点的完整教程
//
// 从零编写一个可注册算子（Scale：y = x * factor）的四步最小流程：
//   1. 用 NodePort 工厂声明端口 Schema
//   2. 编写 RunFn：ctx.input<Tensor>() 类型化读取输入（内建空值/类型校验），
//      经 ctx.failure() 报告结构化错误
//   3. EngineRegistry::registerOperator 注册算子
//   4. createOperator 建节点 → 构图 → 经统一 Task 句柄执行（与算子实现解耦）
//
// 端口 Schema 两种等价写法（推荐工厂：类型与 typeSize 单点书写，改型不漏改）：
//   s.inputs = {Node::Port::in<float>("x")};                            // 工厂（推荐）
//   s.inputs = {{"x", Tensor::TensorType::Float, sizeof(float), {}}};   // 聚合初始化
//
// 需要带默认值的可选输入时用 Node::Port::optional<float>("name", 1.0f)；
// 形状锚定端口用 Node::Port::anchored<float>("name", "anchorPort")。
//
// 预期输出：3.0 * 2.5 = 7.5

#include "InferGraph.h"
#include "EngineRegistry.h"
#include "Tensor.hpp"

#include <cmath>
#include <iostream>
#include <memory>

using namespace DC;

namespace {

// ── 1. Schema：端口用工厂声明（shape 省略 = 标量）──
Node::Schema scaleSchema() {
	Node::Schema s;
	s.inputs = {Node::Port::in<float>("x")};
	s.outputs = {Node::Port::out<float>("y")};
	return s;
}

// ── 2. RunFn：类型化读取输入 + 结构化失败 ──
// factor 是节点私有参数（闭包捕获）；真实算子可改为引擎配置
// （NodeFactoryParams::engineConfig）或额外输入端口。
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
	// ── 3. 注册自定义算子（轻量级：DC::Tensor only、无引擎钩子）──
	auto& reg = EngineRegistry::instance();
	if (!reg.registerOperator("Scale", scaleSchema(), scaleRunFn(2.5f))) {
		std::cerr << "register failed: operator 'Scale' already exists" << std::endl;
		return 1;
	}

	// ── 4. 建图：createOperator → addNode → 绑定图级输入输出 ──
	InferGraph graph;
	graph.addNode(reg.createOperator("Scale", "scale1"));
	graph.bindInput("in", "scale1", "x");
	graph.bindOutput("out", "scale1", "y");

	// ── 5. 取公开接口并执行：宿主一律走统一任务句柄（自定义算子与执行 API 分层无关）──
	auto api = graph.interface(); // 冻结图并一次性解析 bindInput/bindOutput 别名
	auto task = api.createTask(); // 任务句柄：析构自动释放已终止任务

	auto in = Tensor::Create<float>();
	in = 3.0f;
	task.feed("in", std::move(in)); // 按公开输入别名注入

	auto result = task.run(); // 同步：以全部绑定输出提交并等待终止（submit + wait）
	if (result.status != TaskStatus::Succeeded) {
		std::cerr << "task failed" << std::endl;
		for (const auto& err : result.errors)
			std::cerr << "  " << err.nodeName << ": " << err.message << std::endl;
		return 1;
	}

	auto out = task.takeTensor("out"); // 按公开输出别名消费式取出
	const float value = out.item<float>();
	std::cout << "3.0 * 2.5 = " << value << std::endl;

	return std::fabs(value - 7.5f) < 1e-6f ? 0 : 1;
}
