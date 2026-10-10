// Node 多任务乱序输入/输出 单元测试
#include <atomic>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <thread>

#include "Node.h"
#include "NodeExecutor.h"
#include "NodeException.h"
#include "EngineRegistry.h"
#include "OutputZone.h"

using namespace DC;
using TensorType = DC::Tensor::TensorType;
using Tensor = DC::Tensor;
using Shape = DC::Tensor::Shape;

static Node::Schema scalarAddSchema() {
	Node::Schema s;
	s.inputs = {Node::Port::in<float>("a"), Node::Port::in<float>("b")};
	s.outputs = {Node::Port::out<float>("s")};
	return s;
}

static Node::Schema shapedAddSchema(Shape shape) {
	Node::Schema s;
	s.inputs = {Node::Port::in<float>("a", shape), Node::Port::in<float>("b", shape)};
	s.outputs = {Node::Port::out<float>("s", shape)};
	return s;
}

static Node::Result addRunImpl(Node::RunContext& self) {
	const auto& aNT = self.peek("a");
	const auto& bNT = self.peek("b");
	const auto* a = aNT.as<Tensor>();
	const auto* b = bNT.as<Tensor>();

	auto aData = a->data<float>();
	auto bData = b->data<float>();

	std::vector<float> result(aData.size());
	for (size_t i = 0; i < aData.size(); ++i)
		result[i] = aData[i] + bData[i];

	std::vector<std::byte> bytes(result.size() * sizeof(float));
	std::memcpy(bytes.data(), result.data(), bytes.size());

	self.output("s", Value(std::make_unique<Tensor>(TensorType::Float, sizeof(float), a->shape(),
													Tensor::DataBlock(std::move(bytes)))));

	return self.success();
}

static Value makeScalarNative(float value) {
	auto t = std::make_unique<Tensor>(TensorType::Float, sizeof(float));
	*t = value;
	return Value(std::move(t));
}

static Value makeVectorNative(const std::vector<float>& values) {
	std::vector<std::byte> bytes(values.size() * sizeof(float));
	std::memcpy(bytes.data(), values.data(), bytes.size());
	return Value(std::make_unique<Tensor>(TensorType::Float, sizeof(float),
										  Tensor::Shape{static_cast<int64_t>(values.size())},
										  Tensor::DataBlock(std::move(bytes))));
}

static Value makeIntNative(int value) {
	auto t = std::make_unique<Tensor>(TensorType::Int, sizeof(int));
	*t = value;
	return Value(std::move(t));
}

static Tensor makeVectorFloat(const std::vector<float>& values) {
	std::vector<std::byte> bytes(values.size() * sizeof(float));
	std::memcpy(bytes.data(), values.data(), bytes.size());
	return Tensor(TensorType::Float, sizeof(float), {static_cast<int64_t>(values.size())},
				  Tensor::DataBlock(std::move(bytes)));
}

static int failures = 0;

#define CHECK(cond, msg)                                                                                               \
	do {                                                                                                               \
		if (!(cond)) {                                                                                                 \
			std::cerr << "FAIL: " << msg << std::endl;                                                                 \
			++failures;                                                                                                \
			return;                                                                                                    \
		}                                                                                                              \
	} while (0)

#define CHECK_THROWS(stmt, exType, msg)                                                                                \
	do {                                                                                                               \
		try {                                                                                                          \
			stmt;                                                                                                      \
			std::cerr << "FAIL: " << msg << " (no exception thrown)" << std::endl;                                     \
			++failures;                                                                                                \
			return;                                                                                                    \
		} catch (const exType&) {}                                                                                     \
	} while (0)

#define TEST(name)                                                                                                     \
	std::cout << "Test: " << name << " ... " << std::flush;                                                            \
	[&]()
#define END_TEST()                                                                                                     \
	();                                                                                                                \
	std::cout << "PASSED" << std::endl

void runTests() {
	auto& reg = EngineRegistry::instance();

	TEST("out-of-order setInput with explicit tryExecute") {
		auto node = reg.createNode("add1", scalarAddSchema(), addRunImpl);
				NodeExecutor exec(*node);

		std::atomic<bool> completed{false};
		float resultValue = 0.0f;

		node->setCompletionCallback([&](const Node::TaskId& taskId, const Node::Result& result) {
			CHECK(result.ok(), "result should be Ok");
			auto outNT = exec.takeOutput(taskId, "s");
			auto* out = outNT.as<Tensor>();
			resultValue = out->item<float>();
			exec.clearTask(taskId);
			completed = true;
		});

		exec.setInput("task1", "a", makeScalarNative(3.0f));
		CHECK(!completed, "should not complete after only 'a'");
		CHECK_THROWS(exec.tryExecute("task1"), NodeException, "tryExecute should throw when not ready");

		exec.setInput("task1", "b", makeScalarNative(4.0f));
		exec.tryExecute("task1");
		CHECK(completed, "should complete after 'b'");
		CHECK(std::abs(resultValue - 7.0f) < 1e-6f, "scalar add mismatch");
	}
	END_TEST();

	TEST("batch setInputs") {
		auto node = reg.createNode("add2", scalarAddSchema(), addRunImpl);
				NodeExecutor exec(*node);

		std::atomic<bool> completed{false};
		float resultValue = 0.0f;

		node->setCompletionCallback([&](const Node::TaskId& taskId, const Node::Result& result) {
			CHECK(result.ok(), "result should be Ok");
			auto outNT = exec.takeOutput(taskId, "s");
			auto* out = outNT.as<Tensor>();
			resultValue = out->item<float>();
			exec.clearTask(taskId);
			completed = true;
		});

		std::unordered_map<std::string, Node::TaskData> inputs;
		inputs.emplace("a", makeScalarNative(10.0f));
		inputs.emplace("b", makeScalarNative(20.0f));

		exec.setInput("task1", std::move(inputs));
		exec.tryExecute("task1");
		CHECK(completed, "should execute after tryExecute");
		CHECK(std::abs(resultValue - 30.0f) < 1e-6f, "batch add mismatch");
	}
	END_TEST();

	TEST("multi-task interleaving") {
		auto node = reg.createNode("add3", scalarAddSchema(), addRunImpl);
				NodeExecutor exec(*node);

		std::vector<std::string> completedTasks;

		node->setCompletionCallback([&](const Node::TaskId& taskId, const Node::Result& result) {
			CHECK(result.ok(), "result should be Ok");
			completedTasks.push_back(taskId);
			exec.clearTask(taskId);
		});

		exec.setInput("task1", "a", makeScalarNative(1.0f));
		exec.setInput("task2", "b", makeScalarNative(6.0f));
		exec.setInput("task1", "b", makeScalarNative(2.0f));
		exec.tryExecute("task1");

		CHECK(completedTasks.size() == 1, "task1 should complete");
		CHECK(completedTasks[0] == "task1", "task1 should complete first");

		exec.setInput("task2", "a", makeScalarNative(5.0f));
		exec.tryExecute("task2");
		CHECK(completedTasks.size() == 2, "task2 should also complete");
	}
	END_TEST();

	TEST("not ready - no execution") {
		auto node = reg.createNode("add4", scalarAddSchema(), addRunImpl);
				NodeExecutor exec(*node);

		std::atomic<bool> completed{false};
		node->setCompletionCallback([&](const Node::TaskId&, const Node::Result&) { completed = true; });

		exec.setInput("task1", "a", makeScalarNative(1.0f));
		CHECK(!exec.isReady("task1"), "should not be ready with only one input");
		CHECK_THROWS(exec.tryExecute("task1"), NodeException, "tryExecute should throw when not ready");
		CHECK(!completed, "should not execute with only one input");
		CHECK(exec.taskCount() == 1, "one pending task");
	}
	END_TEST();

	TEST("default value unblocks") {
		auto schema = []() {
			Node::Schema s;
			s.inputs = {Node::Port::in<float>("a"),
						Node::Port::optional<float>("b", 100.0f)};
			s.outputs = {Node::Port::out<float>("s")};
			return s;
		}();
		CHECK(schema.valid(), "schema with default should be valid");

		auto node = reg.createNode("add5", schema, addRunImpl);
				NodeExecutor exec(*node);

		std::atomic<bool> completed{false};
		float resultValue = 0.0f;

		node->setCompletionCallback([&](const Node::TaskId& taskId, const Node::Result& result) {
			completed = true;
			if (result.ok()) {
				auto outNT = exec.takeOutput(taskId, "s");
				auto* out = outNT.as<Tensor>();
				resultValue = out->item<float>();
			}
			exec.clearTask(taskId);
		});

		exec.setInput("task1", "a", makeScalarNative(5.0f));
		CHECK(exec.isReady("task1"), "task should be ready with default value");
		exec.tryExecute("task1");
		CHECK(completed, "should execute with default value");
		CHECK(std::abs(resultValue - 105.0f) < 1e-6f, "default value add mismatch");
	}
	END_TEST();

	TEST("default value overridden") {
		auto schema = []() {
			Node::Schema s;
			s.inputs = {Node::Port::in<float>("a"),
						Node::Port::optional<float>("b", 100.0f)};
			s.outputs = {Node::Port::out<float>("s")};
			return s;
		}();

		auto node = reg.createNode("add6", schema, addRunImpl);
				NodeExecutor exec(*node);

		std::atomic<bool> completed{false};
		float resultValue = 0.0f;

		node->setCompletionCallback([&](const Node::TaskId& taskId, const Node::Result& result) {
			completed = true;
			if (result.ok()) {
				auto outNT = exec.takeOutput(taskId, "s");
				auto* out = outNT.as<Tensor>();
				resultValue = out->item<float>();
			}
			exec.clearTask(taskId);
		});

		std::unordered_map<std::string, Node::TaskData> inputs;
		inputs.emplace("a", makeScalarNative(5.0f));
		inputs.emplace("b", makeScalarNative(200.0f));

		exec.setInput("task1", std::move(inputs));
		exec.tryExecute("task1");
		CHECK(completed, "should execute");
		CHECK(std::abs(resultValue - 205.0f) < 1e-6f, "overridden default add mismatch");
	}
	END_TEST();

	TEST("RunFn exception handled") {
		auto schema = []() {
			Node::Schema s;
			s.inputs = {Node::Port::in<float>("x")};
			s.outputs = {Node::Port::out<float>("y")};
			return s;
		}();

		auto node = reg.createNode("thrower", schema,
								   [](Node::RunContext&) -> Node::Result { throw std::runtime_error("boom!"); });
		NodeExecutor exec(*node);

		std::atomic<bool> completed{false};
		Node::Status lastStatus = Node::Status::Ok;

		node->setCompletionCallback([&](const Node::TaskId& taskId, const Node::Result& result) {
			completed = true;
			lastStatus = result.status;
			exec.clearTask(taskId);
		});

		exec.setInput("task1", "x", makeScalarNative(1.0f));
		exec.tryExecute("task1");
		CHECK(completed, "callback should be invoked even on failure");
		CHECK(lastStatus == Node::Status::ExecutionFailed, "status should be ExecutionFailed");
	}
	END_TEST();

	TEST("RunFn missing output") {
		auto schema = []() {
			Node::Schema s;
			s.inputs = {Node::Port::in<float>("x")};
			s.outputs = {Node::Port::out<float>("y")};
			return s;
		}();

		auto node = reg.createNode("bad", schema, [](Node::RunContext& self) -> Node::Result {
			return self.success();
		});
		NodeExecutor exec(*node);

		std::atomic<bool> completed{false};
		Node::Status lastStatus = Node::Status::Ok;

		node->setCompletionCallback([&](const Node::TaskId& taskId, const Node::Result& result) {
			completed = true;
			lastStatus = result.status;
			exec.clearTask(taskId);
		});

		exec.setInput("task1", "x", makeScalarNative(1.0f));
		exec.tryExecute("task1");
		CHECK(completed, "callback should be invoked");
		CHECK(lastStatus == Node::Status::InternalError, "status should be InternalError for missing output");
	}
	END_TEST();

	TEST("polling without callback") {
		auto node = reg.createNode("add9", scalarAddSchema(), addRunImpl);
				NodeExecutor exec(*node);

		exec.setInput("task1", "a", makeScalarNative(7.0f));
		exec.setInput("task1", "b", makeScalarNative(8.0f));
		exec.tryExecute("task1");

		CHECK(exec.hasOutput("task1", "s"), "hasOutput should be true");
		// 并集口径：输入已擦除、输出未取的任务仍计入
		CHECK(exec.taskCount() == 1, "task with undrained outputs must still be counted");
		auto outNT = exec.takeOutput("task1", "s");
		auto* out = outNT.as<Tensor>();
		CHECK(std::abs(out->item<float>() - 15.0f) < 1e-6f, "polling value mismatch");

		CHECK(!exec.hasOutput("task1", "s"), "after takeOutput, hasOutput should be false");
		exec.clearTask("task1");
		CHECK(exec.taskCount() == 0, "task should be cleaned up");
	}
	END_TEST();

	TEST("invalid port name") {
		auto node = reg.createNode("add10", scalarAddSchema(), addRunImpl);
				NodeExecutor exec(*node);

		CHECK_THROWS(exec.setInput("task1", "no_such_port", makeScalarNative(1.0f)), NodeException,
					 "setInput should throw for invalid port");
		CHECK(exec.taskCount() == 0, "no task should be created for invalid port");
	}
	END_TEST();

	TEST("type mismatch rejected at tryExecute") {
		auto node = reg.createNode("add11", scalarAddSchema(), addRunImpl);
				NodeExecutor exec(*node);

		exec.setInput("task1", "b", makeScalarNative(1.0f));
		exec.setInput("task1", "a", makeIntNative(42));

		CHECK(exec.isReady("task1"), "task should appear ready");

		bool threw = false;
		try {
			exec.tryExecute("task1");
		} catch (const NodeException& e) {
			threw = (e.getErrorType() == NodeException::ErrorType::TypeMismatch);
		}
		CHECK(threw, "tryExecute should throw NodeException::TypeMismatch");
	}
	END_TEST();

	TEST("duplicate setInput overwrites") {
		auto node = reg.createNode("add12", scalarAddSchema(), addRunImpl);
				NodeExecutor exec(*node);

		std::atomic<int> callCount{0};
		float resultValue = 0.0f;

		node->setCompletionCallback([&](const Node::TaskId& taskId, const Node::Result& result) {
			++callCount;
			if (result.ok()) {
				auto outNT = exec.takeOutput(taskId, "s");
				auto* out = outNT.as<Tensor>();
				resultValue = out->item<float>();
			}
			exec.clearTask(taskId);
		});

		exec.setInput("task1", "a", makeScalarNative(1.0f));
		exec.setInput("task1", "a", makeScalarNative(10.0f));
		exec.setInput("task1", "b", makeScalarNative(2.0f));
		exec.tryExecute("task1");

		CHECK(callCount == 1, "should execute exactly once");
		CHECK(std::abs(resultValue - 12.0f) < 1e-6f, "should use latest value");
	}
	END_TEST();

	TEST("setInputs fails on invalid port") {
		auto node = reg.createNode("add13", scalarAddSchema(), addRunImpl);
				NodeExecutor exec(*node);

		std::atomic<bool> completed{false};
		node->setCompletionCallback([&](const Node::TaskId&, const Node::Result&) { completed = true; });

		exec.setInput("task1", "a", makeScalarNative(1.0f));

		std::unordered_map<std::string, Node::TaskData> inputs;
		inputs.emplace("b", makeScalarNative(2.0f));
		inputs.emplace("no_such", makeScalarNative(3.0f));

		CHECK_THROWS(exec.setInput("task1", std::move(inputs)), NodeException,
					 "setInputs should throw on invalid port");
		CHECK(!completed, "should not execute after failed setInputs");

		exec.setInput("task1", "b", makeScalarNative(5.0f));
		CHECK(exec.isReady("task1"), "task should be ready");
		exec.tryExecute("task1");
		CHECK(completed, "task should still be executable after rollback");
	}
	END_TEST();

	TEST("callback takeOutput and clearTask") {
		auto node = reg.createNode("add14", scalarAddSchema(), addRunImpl);
				NodeExecutor exec(*node);

		std::atomic<bool> gotOutput{false};
		std::atomic<bool> cleared{false};

		node->setCompletionCallback([&](const Node::TaskId& taskId, const Node::Result& result) {
			CHECK(result.ok(), "result should be Ok");
			auto outNT = exec.takeOutput(taskId, "s");
			auto* out = outNT.as<Tensor>();
			gotOutput = (std::abs(out->item<float>() - 9.0f) < 1e-6f);
			exec.clearTask(taskId);
			cleared = true;
		});

		exec.setInput("task1", "a", makeScalarNative(4.0f));
		exec.setInput("task1", "b", makeScalarNative(5.0f));
		exec.tryExecute("task1");

		CHECK(gotOutput, "callback should get correct output");
		CHECK(cleared, "callback should clear task");
	}
	END_TEST();

	TEST("callback sets input without re-entrant execution") {
		auto node = reg.createNode("add15", scalarAddSchema(), addRunImpl);
				NodeExecutor exec(*node);

		std::atomic<int> callCount{0};
		bool needsTask2{false};

		node->setCompletionCallback([&](const Node::TaskId& taskId, const Node::Result& result) {
			++callCount;
			CHECK(result.ok(), "result should be Ok");
			exec.clearTask(taskId);

			if (callCount == 1) {
				exec.setInput("task2", "a", makeScalarNative(1.0f));
				exec.setInput("task2", "b", makeScalarNative(1.0f));
				needsTask2 = true;
			}
		});

		exec.setInput("task1", "a", makeScalarNative(3.0f));
		exec.setInput("task1", "b", makeScalarNative(3.0f));
		exec.tryExecute("task1");

		CHECK(needsTask2, "callback should have set up task2");
		CHECK(callCount == 1, "only task1 completed so far");
		exec.tryExecute("task2");

		CHECK(callCount == 2, "should process both tasks");
	}
	END_TEST();

	TEST("multi-task interleaving 2") {
		auto node = reg.createNode("add16", scalarAddSchema(), addRunImpl);
				NodeExecutor exec(*node);

		std::atomic<int> callCount{0};

		node->setCompletionCallback([&](const Node::TaskId& taskId, const Node::Result& result) {
			++callCount;
			CHECK(result.ok(), "result should be Ok");
			exec.clearTask(taskId);
		});

		exec.setInput("task1", "a", makeScalarNative(1.0f));
		exec.setInput("task2", "a", makeScalarNative(10.0f));
		exec.setInput("task1", "b", makeScalarNative(2.0f));
		exec.setInput("task2", "b", makeScalarNative(20.0f));
		exec.tryExecute("task1");
		exec.tryExecute("task2");

		CHECK(callCount == 2, "both tasks should complete");
	}
	END_TEST();

	TEST("vector addition with task API") {
		std::vector<float> aVals = {1, 2, 3, 4};
		std::vector<float> bVals = {5, 6, 7, 8};
		std::vector<float> exp = {6, 8, 10, 12};

		auto node = reg.createNode("add17", shapedAddSchema({4}), addRunImpl);
				NodeExecutor exec(*node);

		std::atomic<bool> completed{false};
		bool match = false;

		node->setCompletionCallback([&](const Node::TaskId& taskId, const Node::Result& result) {
			CHECK(result.ok(), "result should be Ok");
			auto outNT = exec.takeOutput(taskId, "s");
			auto* out = outNT.as<Tensor>();
			auto outData = out->data<float>();
			match = true;
			for (size_t i = 0; i < exp.size(); ++i) {
				if (std::abs(outData[i] - exp[i]) > 1e-6f)
					match = false;
			}
			exec.clearTask(taskId);
			completed = true;
		});

		exec.setInput("task1", "a", makeVectorNative(aVals));
		exec.setInput("task1", "b", makeVectorNative(bVals));
		exec.tryExecute("task1");

		CHECK(completed, "vector task should complete");
		CHECK(match, "vector add values should match");
	}
	END_TEST();

	TEST("invalid schema rejected at Node construction (duplicate port names)") {
		Node::Schema s;
		s.inputs = {Node::Port::in<float>("a"), Node::Port::in<float>("a")};
		s.outputs = {Node::Port::out<float>("s")};
		CHECK_THROWS(Node("t", "dup", s, addRunImpl), NodeException,
					 "duplicate port names must be rejected at construction");
	}
	END_TEST();

	TEST("invalid schema rejected at Node construction (typeSize=0 on non-Void port)") {
		Node::Schema s;
		auto p = Node::Port::in<float>("a");
		p.typeSize = 0;
		s.inputs = {p};
		s.outputs = {Node::Port::out<float>("s")};
		CHECK_THROWS(Node("t", "ts0", s, addRunImpl), NodeException,
					 "typeSize=0 on non-Void port must be rejected at construction");
	}
	END_TEST();

	// GraphStore 双保险校验依赖 Node 构造门，不变量由上方构造期用例覆盖

	TEST("typeSize mismatch rejected at execution") {
		auto node = reg.createNode("addTS", scalarAddSchema(), addRunImpl);
		NodeExecutor exec(*node);
		exec.setInput("t1", "a", makeScalarNative(1.0f));
		auto wide = std::make_unique<Tensor>(TensorType::Float, sizeof(double));
		*wide = 2.0;
		exec.setInput("t1", "b", Value(std::move(wide)));
		CHECK_THROWS(exec.tryExecute("t1"), NodeException,
					 "4-byte port must reject 8-byte tensor (typeSize mismatch)");
	}
	END_TEST();

	TEST("output declaration count=0 rejected") {
		OutputZone zone;
		CHECK_THROWS(zone.declare("t1", "n", "y", 0), GraphException,
					 "count=0 declaration must be rejected (omit instead)");
		std::vector<OutputDeclaration> batch = {{"n", "y", 0}, {"n", "z", 1}};
		CHECK_THROWS(zone.declare("t1", std::move(batch)), GraphException,
					 "batch declaration containing count=0 must be rejected atomically");
		CHECK(!zone.hasDeclaration("t1"), "rejected declarations must leave no residue");
		zone.declare("t2", "n", "y", 1);
		CHECK(zone.hasDeclaration("t2"), "count=1 declaration must be accepted");
	}
	END_TEST();

	std::cout << "\nAll Node tests passed!" << std::endl;
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
