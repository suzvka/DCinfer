// 输出激活语义：输出声明按目标节点 kill cone 的当前信号激活状态参与满足判定。
// dead(n) = gated(n) OR ∃ 必需无默认值生产者 p: dead(p)；不活跃声明被跳过，
// 全不活跃 = 空满足 = Succeeded（不揣测宿主意图）。判定每次现读信号，翻转双向生效。
#include "TestHarness.h"
#include "Node.h"
#include "Tensor.hpp"
#include "SignalStore.h"
#include "SignalProbe.h"
#include "CompiledGraph.h"

#include <algorithm>
#include <future>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace DC;

namespace {

void require(bool cond, const std::string& msg) {
	if (!cond)
		throw std::runtime_error(msg);
}

Value makeFloatTensor(float v) {
	auto t = std::make_unique<Tensor>(Tensor::TensorType::Float, sizeof(float));
	*t = v;
	return Value(std::move(t));
}

Node::Schema identitySchema() {
	Node::Schema s;
	s.inputs = {Node::Port::in<float>("x")};
	s.outputs = {Node::Port::out<float>("y")};
	return s;
}

Node::Result identityRun(Node::RunContext& ctx) {
	const auto* in = ctx.peek("x").as<Tensor>();
	if (!in)
		return ctx.failure(Node::Status::InvalidInput, "missing x");
	auto t = std::make_unique<Tensor>(Tensor::TensorType::Float, sizeof(float));
	*t = in->item<float>();
	ctx.output("y", Value(std::move(t)));
	return ctx.success();
}

Node::Schema sumSchema() {
	Node::Schema s;
	s.inputs = {Node::Port::in<float>("a"), Node::Port::in<float>("b")};
	s.outputs = {Node::Port::out<float>("s")};
	return s;
}

Node::Result sumRun(Node::RunContext& ctx) {
	const auto* a = ctx.peek("a").as<Tensor>();
	const auto* b = ctx.peek("b").as<Tensor>();
	if (!a || !b)
		return ctx.failure(Node::Status::InvalidInput, "missing a/b");
	auto t = std::make_unique<Tensor>(Tensor::TensorType::Float, sizeof(float));
	*t = a->item<float>() + b->item<float>();
	ctx.output("s", Value(std::move(t)));
	return ctx.success();
}

// 必需口 req + 带默认值可选口 opt：opt 的生产者被关不致本节点于死地。
Node::Schema defaultSchema() {
	Node::Schema s;
	s.inputs = {Node::Port::in<float>("req"), Node::Port::optional<float>("opt", 0.0f)};
	s.outputs = {Node::Port::out<float>("y")};
	return s;
}

Node::Result defaultRun(Node::RunContext& ctx) {
	const auto* req = ctx.peek("req").as<Tensor>();
	if (!req)
		return ctx.failure(Node::Status::InvalidInput, "missing req");
	float v = req->item<float>();
	if (const auto* opt = ctx.peek("opt").as<Tensor>())
		v += opt->item<float>();
	auto t = std::make_unique<Tensor>(Tensor::TensorType::Float, sizeof(float));
	*t = v;
	ctx.output("y", Value(std::move(t)));
	return ctx.success();
}

bool coneContains(const std::vector<const Node*>& cone, const std::string& name) {
	return std::any_of(cone.begin(), cone.end(),
					   [&](const Node* n) { return n && n->name() == name; });
}

} // namespace

static void runTests() {
	// 1. 门控下游输出被豁免：不再悬空，Succeeded 且无产出。
	{
		TestHarness h;
		h.addNode(std::make_unique<Node>("Builtin", "src", identitySchema(), identityRun));
		auto out = std::make_unique<Node>("Builtin", "out", identitySchema(), identityRun);
		out->bindSignal(h.signalStore(), "g");
		h.addNode(std::move(out));
		h.connect("src", "y", "out", "x");
		h.bindOutput("y", "out", "y");
		h.setSignal("g", false);
		h.feedInput("t", "src", "x", makeFloatTensor(1.0f));
		h.submit("t", "out", "y");
		require(h.awaitCompletion("t"), "gated downstream declaration must not hang");
		require(h.graph().taskStatus("t") == TaskStatus::Succeeded, "waived declaration => Succeeded");
		require(!h.hasOutput("t", "out", "y"), "gated out must not produce");
	}
	std::cout << "Test passed: gated downstream output waived (no hang)" << std::endl;

	// 2. 条件分支：门绑分支头、声明绑分支尾——探针自身门控的盲区，由 cone 覆盖。
	{
		TestHarness h;
		h.addNode(std::make_unique<Node>("Builtin", "src", identitySchema(), identityRun));
		auto head = std::make_unique<Node>("Builtin", "head", identitySchema(), identityRun);
		head->bindSignal(h.signalStore(), "g");
		h.addNode(std::move(head));
		h.addNode(std::make_unique<Node>("Builtin", "tail", identitySchema(), identityRun));
		h.connect("src", "y", "head", "x");
		h.connect("head", "y", "tail", "x");
		h.bindOutput("y", "tail", "y");
		h.setSignal("g", false);
		h.feedInput("t", "src", "x", makeFloatTensor(1.0f));
		h.submit("t", "tail", "y");
		require(h.awaitCompletion("t"), "gated branch must not hang on tail declaration");
		require(h.graph().taskStatus("t") == TaskStatus::Succeeded, "gated branch => Succeeded");
		require(!h.hasOutput("t", "tail", "y"), "tail behind gated head must not produce");
	}
	std::cout << "Test passed: gate at branch head waives tail declaration" << std::endl;

	// 3. 正向对照：门开 → 分支尾正常产出。
	{
		TestHarness h;
		h.addNode(std::make_unique<Node>("Builtin", "src", identitySchema(), identityRun));
		auto head = std::make_unique<Node>("Builtin", "head", identitySchema(), identityRun);
		head->bindSignal(h.signalStore(), "g");
		h.addNode(std::move(head));
		h.addNode(std::make_unique<Node>("Builtin", "tail", identitySchema(), identityRun));
		h.connect("src", "y", "head", "x");
		h.connect("head", "y", "tail", "x");
		h.bindOutput("y", "tail", "y");
		h.setSignal("g", true);
		h.feedInput("t", "src", "x", makeFloatTensor(7.0f));
		h.submit("t", "tail", "y");
		require(h.awaitCompletion("t"), "open branch should complete");
		require(h.hasOutput("t", "tail", "y"), "tail should produce when gate open");
		require(h.getOutputTensor("t", "tail", "y").item<float>() == 7.0f, "identity value preserved");
	}
	std::cout << "Test passed: open gate produces declared output" << std::endl;

	// 4. 扇入 AND：一个必需生产者被关 → 消费者输入永凑不齐 → 豁免。
	//    任意路径可达 BFS 会因另一活路误判活跃而悬空；cone 沿必需边传播死区，正确豁免。
	{
		TestHarness h;
		h.addNode(std::make_unique<Node>("Builtin", "src", identitySchema(), identityRun));
		auto p1 = std::make_unique<Node>("Builtin", "p1", identitySchema(), identityRun);
		p1->bindSignal(h.signalStore(), "g");
		h.addNode(std::move(p1));
		h.addNode(std::make_unique<Node>("Builtin", "p2", identitySchema(), identityRun));
		h.addNode(std::make_unique<Node>("Builtin", "c", sumSchema(), sumRun));
		h.connect("src", "y", "p1", "x");
		h.connect("src", "y", "p2", "x");
		h.connect("p1", "y", "c", "a");
		h.connect("p2", "y", "c", "b");
		h.bindOutput("s", "c", "s");
		h.setSignal("g", false);
		h.feedInput("t", "src", "x", makeFloatTensor(3.0f));
		h.submit("t", "c", "s");
		require(h.awaitCompletion("t"), "fan-in with gated required producer must not hang");
		require(h.graph().taskStatus("t") == TaskStatus::Succeeded, "AND-dead consumer => Succeeded");
		require(!h.hasOutput("t", "c", "s"), "consumer missing required input must not produce");
	}
	std::cout << "Test passed: fan-in AND propagates deadness from gated required producer" << std::endl;

	// 5a. 默认值口不传播死区（行为）：被关生产者接入默认口，消费者仍从必需口运行产出。
	{
		TestHarness h;
		auto p = std::make_unique<Node>("Builtin", "p", identitySchema(), identityRun);
		p->bindSignal(h.signalStore(), "g");
		h.addNode(std::move(p));
		h.addNode(std::make_unique<Node>("Builtin", "c", defaultSchema(), defaultRun));
		h.connect("p", "y", "c", "opt");
		h.bindOutput("y", "c", "y");
		h.setSignal("g", false);
		h.feedInput("t", "c", "req", makeFloatTensor(5.0f));
		h.submit("t", "c", "y");
		require(h.awaitCompletion("t"), "consumer with defaulted gated input should complete");
		require(h.graph().taskStatus("t") == TaskStatus::Succeeded, "defaulted port keeps consumer active");
		require(h.hasOutput("t", "c", "y"), "consumer runs from required input");
		require(h.getOutputTensor("t", "c", "y").item<float>() == 5.0f, "default opt contributes 0");
	}
	std::cout << "Test passed: defaulted input port does not propagate deadness (behavior)" << std::endl;

	// 5b. killCone 单测：必需边入锥、默认/可选边不入锥、含自身。
	{
		TestHarness h;
		h.addNode(std::make_unique<Node>("Builtin", "src", identitySchema(), identityRun));
		h.addNode(std::make_unique<Node>("Builtin", "mid", identitySchema(), identityRun));
		h.addNode(std::make_unique<Node>("Builtin", "out", defaultSchema(), defaultRun));
		h.addNode(std::make_unique<Node>("Builtin", "optSrc", identitySchema(), identityRun));
		h.connect("src", "y", "mid", "x");
		h.connect("mid", "y", "out", "req");
		h.connect("optSrc", "y", "out", "opt");
		auto snap = h.graph().freeze();
		auto cone = killCone(snap->runtimeView(), "out");
		require(coneContains(cone, "out"), "cone includes target itself");
		require(coneContains(cone, "mid"), "required producer in cone");
		require(coneContains(cone, "src"), "transitive required producer in cone");
		require(!coneContains(cone, "optSrc"), "defaulted-port producer excluded from cone");
	}
	std::cout << "Test passed: killCone includes required chain, excludes defaulted port" << std::endl;

	// 6. 全豁免 = 空满足 = Succeeded，无 Error 诊断（不揣测意图）。
	{
		TestHarness h;
		h.addNode(std::make_unique<Node>("Builtin", "src", identitySchema(), identityRun));
		auto b1 = std::make_unique<Node>("Builtin", "b1", identitySchema(), identityRun);
		b1->bindSignal(h.signalStore(), "g1");
		h.addNode(std::move(b1));
		auto b2 = std::make_unique<Node>("Builtin", "b2", identitySchema(), identityRun);
		b2->bindSignal(h.signalStore(), "g2");
		h.addNode(std::move(b2));
		h.connect("src", "y", "b1", "x");
		h.connect("src", "y", "b2", "x");
		h.bindOutput("y1", "b1", "y");
		h.bindOutput("y2", "b2", "y");
		h.setSignal("g1", false);
		h.setSignal("g2", false);
		h.feedInput("t", "src", "x", makeFloatTensor(1.0f));
		h.submit("t", std::vector<OutputDeclaration>{{"b1", "y", 1}, {"b2", "y", 1}});
		require(h.awaitCompletion("t"), "all-waived declarations must terminate");
		require(h.graph().taskStatus("t") == TaskStatus::Succeeded, "all waived => vacuous Succeeded");
		require(!h.hasOutput("t", "b1", "y") && !h.hasOutput("t", "b2", "y"), "no gated branch produces");
	}
	std::cout << "Test passed: all-waived declarations => vacuous Succeeded" << std::endl;

	// 7. 中途翻转：submit 时门开，在飞期间关门；判定现读当前信号 → 豁免终止。
	//    若按 submit 快照判定，out 恒活跃 → 声明永不满足 → 悬空超时。
	{
		auto started = std::make_shared<std::promise<void>>();
		auto release = std::make_shared<std::promise<void>>();
		auto startedFut = started->get_future();
		auto releaseFut = std::make_shared<std::shared_future<void>>(release->get_future().share());
		auto slowRun = [started, releaseFut](Node::RunContext& ctx) -> Node::Result {
			const auto* in = ctx.peek("x").as<Tensor>();
			if (!in)
				return ctx.failure(Node::Status::InvalidInput, "missing x");
			float v = in->item<float>();
			started->set_value();
			releaseFut->wait();
			auto t = std::make_unique<Tensor>(Tensor::TensorType::Float, sizeof(float));
			*t = v;
			ctx.output("y", Value(std::move(t)));
			return ctx.success();
		};
		TestHarness h;
		h.addNode(std::make_unique<Node>("Builtin", "src", identitySchema(), slowRun));
		auto out = std::make_unique<Node>("Builtin", "out", identitySchema(), identityRun);
		out->bindSignal(h.signalStore(), "g");
		h.addNode(std::move(out));
		h.connect("src", "y", "out", "x");
		h.bindOutput("y", "out", "y");
		h.setSignal("g", true);
		h.feedInput("t", "src", "x", makeFloatTensor(9.0f));
		h.submit("t", "out", "y");
		startedFut.wait();           // src 已进入 RunFn，任务在飞
		h.setSignal("g", false);     // 中途关门
		release->set_value();        // 放行 src 完成
		require(h.awaitCompletion("t"), "mid-flight waiver must terminate (not snapshot-based)");
		require(h.graph().taskStatus("t") == TaskStatus::Succeeded, "mid-flight gated => Succeeded");
		require(!h.hasOutput("t", "out", "y"), "out gated at check time => waived, no output");
	}
	std::cout << "Test passed: mid-flight signal flip waives declaration dynamically" << std::endl;

	// 8. 无信号图兼容：不登记探针，行为与升级前一致。
	{
		TestHarness h;
		h.addNode(std::make_unique<Node>("Builtin", "src", identitySchema(), identityRun));
		h.addNode(std::make_unique<Node>("Builtin", "out", identitySchema(), identityRun));
		h.connect("src", "y", "out", "x");
		h.bindOutput("y", "out", "y");
		h.feedInput("t", "src", "x", makeFloatTensor(2.0f));
		h.submit("t", "out", "y");
		require(h.awaitCompletion("t"), "signal-free graph should complete");
		require(h.graph().taskStatus("t") == TaskStatus::Succeeded, "signal-free => Succeeded");
		require(h.getOutputTensor("t", "out", "y").item<float>() == 2.0f, "signal-free output preserved");
	}
	std::cout << "Test passed: signal-free graph unchanged" << std::endl;

	std::cout << "\nAll OutputActivation tests passed!" << std::endl;
}

int main() {
	try {
		runTests();
		return 0;
	} catch (const std::exception& e) {
		std::cerr << "Test failure: " << e.what() << std::endl;
		return -1;
	}
}