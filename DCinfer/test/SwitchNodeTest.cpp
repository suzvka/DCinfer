// switch_node_test.cpp - SwitchNode integration test
//
// Verifies:
//   1. Basic switch: 3 candidates (mul 1/2/3), runtime select changes output
//   2. Graph integration: SwitchNode in a src->switch->dst pipeline
//   3. Out-of-range protection: active index beyond candidates returns error
//   4. Node type identity: SwitchNode reports type='Switch'

#include "SwitchNode.h"
#include "InferGraph.h"
#include "Tensor.hpp"

#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

#define CHECK(cond, msg)                                                       \
	do {                                                                       \
		if (!(cond)) {                                                         \
			std::cerr << "[FAIL] " << (msg) << "  (" << __FILE__ << ":"        \
					  << __LINE__ << ")" << std::endl;                         \
			++g_failures;                                                      \
		}                                                                      \
	} while (0)

#define REPORT()                                                               \
	do {                                                                       \
		if (g_failures == 0)                                                   \
			std::cout << "  -> PASS" << std::endl;                             \
		else {                                                                 \
			std::cout << "  -> FAILED" << std::endl;                           \
			return;                                                            \
		}                                                                      \
	} while (0)

// Helper: create a scalar float Value (for feedInput)
DC::Value makeFloatTensor(float v) {
	auto t = std::make_unique<DC::Tensor>(DC::Tensor::TensorType::Float, sizeof(float));
	*t = v;
	return DC::Value(std::move(t));
}

// Helper: make a RunFn that reads 'x', multiplies by factor, writes to 'y'
DC::Node::RunFn makeMulRunFn(float factor) {
	return [factor](DC::Node::RunContext& ctx) -> DC::Node::Result {
		const auto& val = ctx.peek("x");
		const auto* t = val.as<DC::Tensor>();
		if (!t)
			return ctx.failure(DC::Node::Status::InvalidInput, "expected Tensor");
		float result = t->item<float>() * factor;
		auto out = std::make_unique<DC::Tensor>(DC::Tensor::TensorType::Float, sizeof(float));
		*out = result;
		ctx.output("y", DC::Value(std::move(out)));
		return ctx.success();
	};
}

float readScalar(const DC::Tensor& t) {
	return t.item<float>();
}

// ════════════════════════════════════════════
// Test 1: Basic switch (3 candidates)
// ════════════════════════════════════════════
void testBasicSwitch() {
	std::cout << "Test 1: Basic switch (mul1, mul2, mul3)" << std::endl;

	DC::Node::Schema schema;
	schema.inputs = {DC::Node::Port::in<float>("x", {1})};
	schema.outputs = {DC::Node::Port::out<float>("y", {1})};

	std::vector<DC::SwitchCandidate> candidates = {
		{"mul1", makeMulRunFn(1.0f), nullptr, nullptr},
		{"mul2", makeMulRunFn(2.0f), nullptr, nullptr},
		{"mul3", makeMulRunFn(3.0f), nullptr, nullptr},
	};

	auto [node, handle] = DC::createSwitchNode("sw", std::move(schema), std::move(candidates));
	CHECK(handle.current() == 0, "initial index should be 0");

	DC::InferGraph graph;
	graph.addNode(std::move(node));
	graph.bindOutput("sw", "y");

	// cand 0 (mul1): 5.0 -> 5.0
	std::optional<DC::Tensor> captured;
	graph.setTaskCompleteCallback([&](const DC::InferGraph::TaskId& tid) {
		if (graph.hasOutput(tid, "sw", "y")) {
			auto val = graph.takeOutput(tid, "sw", "y");
			if (auto* t = val.as<DC::Tensor>())
				captured = std::move(*t);
		}
	});
	graph.feedInput("t1", "sw", "x", makeFloatTensor(5.0f));
	graph.submit("t1", "sw", "y");
	CHECK(graph.wait("t1", std::chrono::milliseconds(3000)), "t1 complete");
	CHECK(captured.has_value(), "t1 has output");
	if (captured)
		CHECK(std::abs(readScalar(*captured) - 5.0f) < 1e-5f, "mul1: 5*1=5");

	// switch to cand 1 (mul2): 5.0 -> 10.0
	handle.select(1);
	CHECK(handle.current() == 1, "index is 1");
	captured.reset();
	graph.feedInput("t2", "sw", "x", makeFloatTensor(5.0f));
	graph.submit("t2", "sw", "y");
	CHECK(graph.wait("t2", std::chrono::milliseconds(3000)), "t2 complete");
	if (captured)
		CHECK(std::abs(readScalar(*captured) - 10.0f) < 1e-5f, "mul2: 5*2=10");

	// switch to cand 2 (mul3): 4.0 -> 12.0
	handle.select(2);
	captured.reset();
	graph.feedInput("t3", "sw", "x", makeFloatTensor(4.0f));
	graph.submit("t3", "sw", "y");
	CHECK(graph.wait("t3", std::chrono::milliseconds(3000)), "t3 complete");
	if (captured)
		CHECK(std::abs(readScalar(*captured) - 12.0f) < 1e-5f, "mul3: 4*3=12");

	REPORT();
}

// ════════════════════════════════════════════
// Test 2: Graph integration (src -> switch -> dst)
// ════════════════════════════════════════════
void testGraphIntegration() {
	std::cout << "Test 2: Graph pipeline (src -> switch -> dst)" << std::endl;

	DC::Node::Schema identitySchema;
	identitySchema.inputs = {DC::Node::Port::in<float>("x", {1})};
	identitySchema.outputs = {DC::Node::Port::out<float>("y", {1})};

	auto identityFn = [](DC::Node::RunContext& ctx) -> DC::Node::Result {
		const auto& val = ctx.peek("x");
		const auto* t = val.as<DC::Tensor>();
		if (!t)
			return ctx.failure(DC::Node::Status::InvalidInput, "not Tensor");
		auto out = std::make_unique<DC::Tensor>(*t);
		ctx.output("y", DC::Value(std::move(out)));
		return ctx.success();
	};

	auto addOneFn = [](DC::Node::RunContext& ctx) -> DC::Node::Result {
		const auto& val = ctx.peek("x");
		const auto* t = val.as<DC::Tensor>();
		if (!t)
			return ctx.failure(DC::Node::Status::InvalidInput, "not Tensor");
		float result = t->item<float>() + 1.0f;
		auto out = std::make_unique<DC::Tensor>(DC::Tensor::TensorType::Float, sizeof(float));
		*out = result;
		ctx.output("y", DC::Value(std::move(out)));
		return ctx.success();
	};

	DC::Node::Schema swSchema;
	swSchema.inputs = {DC::Node::Port::in<float>("x", {1})};
	swSchema.outputs = {DC::Node::Port::out<float>("y", {1})};
	std::vector<DC::SwitchCandidate> cands = {
		{"mul5", makeMulRunFn(5.0f), nullptr, nullptr},
		{"mul7", makeMulRunFn(7.0f), nullptr, nullptr},
	};
	auto [swNode, handle] = DC::createSwitchNode("sw", std::move(swSchema), std::move(cands));

	DC::InferGraph graph;
	graph.addNode(std::make_unique<DC::Node>("Builtin", "src", identitySchema, identityFn));
	graph.addNode(std::move(swNode));
	graph.addNode(std::make_unique<DC::Node>("Builtin", "dst", identitySchema, addOneFn));

	graph.connect("src", "y", "sw", "x");
	graph.connect("sw", "y", "dst", "x");
	graph.bindOutput("dst", "y");

	std::optional<DC::Tensor> captured;
	graph.setTaskCompleteCallback([&](const DC::InferGraph::TaskId& tid) {
		if (graph.hasOutput(tid, "dst", "y")) {
			auto val = graph.takeOutput(tid, "dst", "y");
			if (auto* t = val.as<DC::Tensor>())
				captured = std::move(*t);
		}
	});

	// mul5: 3.0 -> id(3) -> mul5(15) -> add1(16)
	graph.feedInput("t1", "src", "x", makeFloatTensor(3.0f));
	graph.submit("t1", "dst", "y");
	CHECK(graph.wait("t1", std::chrono::milliseconds(3000)), "t1 complete");
	CHECK(captured.has_value(), "t1 has output");
	if (captured)
		CHECK(std::abs(readScalar(*captured) - 16.0f) < 1e-5f, "pipeline mul5: (3*5)+1=16");

	// switch to mul7: 3.0 -> id(3) -> mul7(21) -> add1(22)
	handle.select(1);
	captured.reset();
	graph.feedInput("t2", "src", "x", makeFloatTensor(3.0f));
	graph.submit("t2", "dst", "y");
	CHECK(graph.wait("t2", std::chrono::milliseconds(3000)), "t2 complete");
	if (captured)
		CHECK(std::abs(readScalar(*captured) - 22.0f) < 1e-5f, "pipeline mul7: (3*7)+1=22");

	REPORT();
}

// ════════════════════════════════════════════
// Test 3: Out-of-range index (direct tryExecute check)
// ════════════════════════════════════════════
void testOutOfRange() {
	std::cout << "Test 3: Out-of-range index produces error" << std::endl;

	DC::Node::Schema schema;
	schema.inputs = {DC::Node::Port::in<float>("x", {1})};
	schema.outputs = {DC::Node::Port::out<float>("y", {1})};
	std::vector<DC::SwitchCandidate> cands = {
		{"only", makeMulRunFn(1.0f), nullptr, nullptr},
	};
	auto [node, handle] = DC::createSwitchNode("sw_oob", std::move(schema), std::move(cands));

	handle.select(99); // out of range

	// Direct test: feed input and tryExecute without graph
	node->setInput("t1", "x", makeFloatTensor(1.0f));
	CHECK(node->isReady("t1"), "node should be ready");
	auto result = node->tryExecute("t1");
	CHECK(!result.ok(), "out-of-range should fail");
	CHECK(result.message.find("out of range") != std::string::npos, "error contains 'out of range'");

	REPORT();
}

// ════════════════════════════════════════════
// Test 4: Node type identity
// ════════════════════════════════════════════
void testNodeType() {
	std::cout << "Test 4: SwitchNode type='Switch'" << std::endl;

	DC::Node::Schema schema;
	schema.inputs = {DC::Node::Port::in<float>("x", {1})};
	schema.outputs = {DC::Node::Port::out<float>("y", {1})};
	std::vector<DC::SwitchCandidate> cands = {
		{"a", makeMulRunFn(1.0f), nullptr, nullptr},
	};
	auto [node, handle] = DC::createSwitchNode("sw_type", std::move(schema), std::move(cands));
	CHECK(node->type() == "Switch", "type is Switch");
	CHECK(node->name() == "sw_type", "name is sw_type");
	CHECK(node->schema().inputs.size() == 1, "1 input port");
	CHECK(node->schema().outputs.size() == 1, "1 output port");

	REPORT();
}

} // anonymous namespace

int main() {
	try {
		testBasicSwitch();
		testGraphIntegration();
		testOutOfRange();
		testNodeType();

		if (g_failures == 0) {
			std::cout << "\n=== All SwitchNode tests passed! ===" << std::endl;
			return 0;
		} else {
			std::cout << "\n=== " << g_failures << " test(s) FAILED ===" << std::endl;
			return 1;
		}
	} catch (const std::exception& e) {
		std::cerr << "Exception: " << e.what() << std::endl;
		return -1;
	}
}
