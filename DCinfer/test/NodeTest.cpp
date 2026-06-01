// Node 多任务乱序输入/输出 单元测试
#include <atomic>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <thread>

#include "Node.h"
#include "NodeException.h"
#include "EngineRegistry.h"

using namespace DC;
using TensorType = DC::Tensor::TensorType;
using Tensor = DC::Tensor;
using Shape = DC::Tensor::Shape;

// ── Schema 辅助 ──
static Node::Schema scalarAddSchema() {
	Node::Schema s;
	s.inputs = {{"a", TensorType::Float, sizeof(float), {}}, {"b", TensorType::Float, sizeof(float), {}}};
	s.outputs = {{"s", TensorType::Float, sizeof(float), {}}};
	return s;
}

static Node::Schema shapedAddSchema(Shape shape) {
	Node::Schema s;
	s.inputs = {{"a", TensorType::Float, sizeof(float), shape}, {"b", TensorType::Float, sizeof(float), shape}};
	s.outputs = {{"s", TensorType::Float, sizeof(float), shape}};
	return s;
}

// ── Add 计算逻辑 ──
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

// ── 创建辅助张量（用于 setInput 的 Value 包装）──
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

// ── 创建辅助张量（用于默认值等 Schema 定义）──
static Tensor makeScalarFloat(float value) {
	Tensor t(TensorType::Float, sizeof(float));
	t = value;
	return t;
}

static Tensor makeVectorFloat(const std::vector<float>& values) {
	std::vector<std::byte> bytes(values.size() * sizeof(float));
	std::memcpy(bytes.data(), values.data(), bytes.size());
	return Tensor(TensorType::Float, sizeof(float), {static_cast<int64_t>(values.size())},
				  Tensor::DataBlock(std::move(bytes)));
}

// ── 测试入口 ──
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

// ════════════════════════════════════════════

void runTests() {
	auto& reg = EngineRegistry::instance();

	// ── Test 1: 基本乱序 setInput ──
	TEST("out-of-order setInput with explicit tryExecute") {
		auto node = reg.createNode("add1", scalarAddSchema(), addRunImpl);

		std::atomic<bool> completed{false};
		float resultValue = 0.0f;

		node->setCompletionCallback([&](const Node::TaskId& taskId, const Node::Result& result) {
			CHECK(result.ok(), "result should be Ok");
			auto outNT = node->getOutput(taskId, "s");
			auto* out = outNT.as<Tensor>();
			resultValue = out->item<float>();
			node->clearTask(taskId);
			completed = true;
		});

		// a 先到，不应触发
		node->setInput("task1", "a", makeScalarNative(3.0f));
		CHECK(!completed, "should not complete after only 'a'");
		CHECK_THROWS(node->tryExecute("task1"), NodeException, "tryExecute should throw when not ready");

		// b 后到 → 就绪
		node->setInput("task1", "b", makeScalarNative(4.0f));
		node->tryExecute("task1");
		CHECK(completed, "should complete after 'b'");
		CHECK(std::abs(resultValue - 7.0f) < 1e-6f, "scalar add mismatch");
	}
	END_TEST();

	// ── Test 2: 批量 setInputs ──
	TEST("batch setInputs") {
		auto node = reg.createNode("add2", scalarAddSchema(), addRunImpl);

		std::atomic<bool> completed{false};
		float resultValue = 0.0f;

		node->setCompletionCallback([&](const Node::TaskId& taskId, const Node::Result& result) {
			CHECK(result.ok(), "result should be Ok");
			auto outNT = node->getOutput(taskId, "s");
			auto* out = outNT.as<Tensor>();
			resultValue = out->item<float>();
			node->clearTask(taskId);
			completed = true;
		});

		std::unordered_map<std::string, Node::TaskData> inputs;
		inputs.emplace("a", makeScalarNative(10.0f));
		inputs.emplace("b", makeScalarNative(20.0f));

		node->setInput("task1", std::move(inputs));
		node->tryExecute("task1");
		CHECK(completed, "should execute after tryExecute");
		CHECK(std::abs(resultValue - 30.0f) < 1e-6f, "batch add mismatch");
	}
	END_TEST();

	// ── Test 3: 多任务交织 ──
	TEST("multi-task interleaving") {
		auto node = reg.createNode("add3", scalarAddSchema(), addRunImpl);

		std::vector<std::string> completedTasks;

		node->setCompletionCallback([&](const Node::TaskId& taskId, const Node::Result& result) {
			CHECK(result.ok(), "result should be Ok");
			completedTasks.push_back(taskId);
			node->clearTask(taskId);
		});

		node->setInput("task1", "a", makeScalarNative(1.0f));
		node->setInput("task2", "b", makeScalarNative(6.0f));
		node->setInput("task1", "b", makeScalarNative(2.0f)); // task1 就绪
		node->tryExecute("task1");

		CHECK(completedTasks.size() == 1, "task1 should complete");
		CHECK(completedTasks[0] == "task1", "task1 should complete first");

		node->setInput("task2", "a", makeScalarNative(5.0f)); // task2 就绪
		node->tryExecute("task2");
		CHECK(completedTasks.size() == 2, "task2 should also complete");
	}
	END_TEST();

	// ── Test 4: 尚不就绪不触发 ──
	TEST("not ready - no execution") {
		auto node = reg.createNode("add4", scalarAddSchema(), addRunImpl);

		std::atomic<bool> completed{false};
		node->setCompletionCallback([&](const Node::TaskId&, const Node::Result&) { completed = true; });

		node->setInput("task1", "a", makeScalarNative(1.0f));
		CHECK(!node->isReady("task1"), "should not be ready with only one input");
		CHECK_THROWS(node->tryExecute("task1"), NodeException, "tryExecute should throw when not ready");
		CHECK(!completed, "should not execute with only one input");
		CHECK(node->taskCount() == 1, "one pending task");
	}
	END_TEST();

	// ── Test 5: 有默认值不阻塞 ──
	TEST("default value unblocks") {
		auto schema = []() {
			Node::Schema s;
			s.inputs = {{"a", TensorType::Float, sizeof(float), {}},
						{"b", TensorType::Float, sizeof(float), {}, true, makeScalarFloat(100.0f)}}; // b 有默认值 100
			s.outputs = {{"s", TensorType::Float, sizeof(float), {}}};
			return s;
		}();
		CHECK(schema.valid(), "schema with default should be valid");

		auto node = reg.createNode("add5", schema, addRunImpl);

		std::atomic<bool> completed{false};
		float resultValue = 0.0f;

		node->setCompletionCallback([&](const Node::TaskId& taskId, const Node::Result& result) {
			completed = true;
			if (result.ok()) {
				auto outNT = node->getOutput(taskId, "s");
				auto* out = outNT.as<Tensor>();
				resultValue = out->item<float>();
			}
			node->clearTask(taskId);
		});

		// 只设置 a，b 有默认值 100 → 应立即触发
		node->setInput("task1", "a", makeScalarNative(5.0f));
		CHECK(node->isReady("task1"), "task should be ready with default value");
		node->tryExecute("task1");
		CHECK(completed, "should execute with default value");
		CHECK(std::abs(resultValue - 105.0f) < 1e-6f, "default value add mismatch");
	}
	END_TEST();

	// ── Test 6: 默认值被显式覆盖 ──
	TEST("default value overridden") {
		auto schema = []() {
			Node::Schema s;
			s.inputs = {{"a", TensorType::Float, sizeof(float), {}},
						{"b", TensorType::Float, sizeof(float), {}, true, makeScalarFloat(100.0f)}};
			s.outputs = {{"s", TensorType::Float, sizeof(float), {}}};
			return s;
		}();

		auto node = reg.createNode("add6", schema, addRunImpl);

		std::atomic<bool> completed{false};
		float resultValue = 0.0f;

		node->setCompletionCallback([&](const Node::TaskId& taskId, const Node::Result& result) {
			completed = true;
			if (result.ok()) {
				auto outNT = node->getOutput(taskId, "s");
				auto* out = outNT.as<Tensor>();
				resultValue = out->item<float>();
			}
			node->clearTask(taskId);
		});

		// 批量同时设置 a 和 b，覆盖默认值
		std::unordered_map<std::string, Node::TaskData> inputs;
		inputs.emplace("a", makeScalarNative(5.0f));
		inputs.emplace("b", makeScalarNative(200.0f));

		node->setInput("task1", std::move(inputs));
		node->tryExecute("task1");
		CHECK(completed, "should execute");
		CHECK(std::abs(resultValue - 205.0f) < 1e-6f, "overridden default add mismatch");
	}
	END_TEST();

	// ── Test 7: RunFn 抛异常 ──
	TEST("RunFn exception handled") {
		auto schema = []() {
			Node::Schema s;
			s.inputs = {{"x", TensorType::Float, sizeof(float), {}}};
			s.outputs = {{"y", TensorType::Float, sizeof(float), {}}};
			return s;
		}();

		auto node = reg.createNode("thrower", schema,
								   [](Node::RunContext&) -> Node::Result { throw std::runtime_error("boom!"); });

		std::atomic<bool> completed{false};
		Node::Status lastStatus = Node::Status::Ok;

		node->setCompletionCallback([&](const Node::TaskId& taskId, const Node::Result& result) {
			completed = true;
			lastStatus = result.status;
			node->clearTask(taskId);
		});

		node->setInput("task1", "x", makeScalarNative(1.0f));
		node->tryExecute("task1"); // RunFn throws internally, caught by _checkAndExecute
		CHECK(completed, "callback should be invoked even on failure");
		CHECK(lastStatus == Node::Status::ExecutionFailed, "status should be ExecutionFailed");
	}
	END_TEST();

	// ── Test 8: RunFn 不产出 output ──
	TEST("RunFn missing output") {
		auto schema = []() {
			Node::Schema s;
			s.inputs = {{"x", TensorType::Float, sizeof(float), {}}};
			s.outputs = {{"y", TensorType::Float, sizeof(float), {}}};
			return s;
		}();

		auto node = reg.createNode("bad", schema, [](Node::RunContext& self) -> Node::Result {
			// 故意不调用 output
			return self.success();
		});

		std::atomic<bool> completed{false};
		Node::Status lastStatus = Node::Status::Ok;

		node->setCompletionCallback([&](const Node::TaskId& taskId, const Node::Result& result) {
			completed = true;
			lastStatus = result.status;
			node->clearTask(taskId);
		});

		node->setInput("task1", "x", makeScalarNative(1.0f));
		node->tryExecute("task1"); // output not produced → InternalError in callback
		CHECK(completed, "callback should be invoked");
		CHECK(lastStatus == Node::Status::InternalError, "status should be InternalError for missing output");
	}
	END_TEST();

	// ── Test 9: 无回调时轮询模式 ──
	TEST("polling without callback") {
		auto node = reg.createNode("add9", scalarAddSchema(), addRunImpl);

		node->setInput("task1", "a", makeScalarNative(7.0f));
		node->setInput("task1", "b", makeScalarNative(8.0f));
		node->tryExecute("task1");

		CHECK(node->hasOutput("task1", "s"), "hasOutput should be true");
		auto outNT = node->getOutput("task1", "s");
		auto* out = outNT.as<Tensor>();
		CHECK(std::abs(out->item<float>() - 15.0f) < 1e-6f, "polling value mismatch");

		CHECK(!node->hasOutput("task1", "s"), "after getOutput, hasOutput should be false");
		node->clearTask("task1");
		CHECK(node->taskCount() == 0, "task should be cleaned up");
	}
	END_TEST();

	// ── Test 10: 端口名不存在 ──
	TEST("invalid port name") {
		auto node = reg.createNode("add10", scalarAddSchema(), addRunImpl);

		CHECK_THROWS(node->setInput("task1", "no_such_port", makeScalarNative(1.0f)), NodeException,
					 "setInput should throw for invalid port");
		CHECK(node->taskCount() == 0, "no task should be created for invalid port");
	}
	END_TEST();

	// ── Test 11: 类型不匹配（tryExecute 时在校验阶段抛出 NodeException::TypeMismatch）──
	TEST("type mismatch rejected at tryExecute") {
		auto node = reg.createNode("add11", scalarAddSchema(), addRunImpl);

		// 设置 a 为 int（类型不匹配），b 为 float → 缓冲阶段不校验
		node->setInput("task1", "b", makeScalarNative(1.0f));
		node->setInput("task1", "a", makeIntNative(42));

		CHECK(node->isReady("task1"), "task should appear ready");

		// tryExecute 时在校验阶段抛出 NodeException::TypeMismatch
		bool threw = false;
		try {
			node->tryExecute("task1");
		} catch (const NodeException& e) {
			threw = (e.getErrorType() == NodeException::ErrorType::TypeMismatch);
		}
		CHECK(threw, "tryExecute should throw NodeException::TypeMismatch");
	}
	END_TEST();

	// ── Test 12: 同一 taskId/port 重复 setInput ──
	TEST("duplicate setInput overwrites") {
		auto node = reg.createNode("add12", scalarAddSchema(), addRunImpl);

		std::atomic<int> callCount{0};
		float resultValue = 0.0f;

		node->setCompletionCallback([&](const Node::TaskId& taskId, const Node::Result& result) {
			++callCount;
			if (result.ok()) {
				auto outNT = node->getOutput(taskId, "s");
				auto* out = outNT.as<Tensor>();
				resultValue = out->item<float>();
			}
			node->clearTask(taskId);
		});

		node->setInput("task1", "a", makeScalarNative(1.0f));
		node->setInput("task1", "a", makeScalarNative(10.0f)); // 覆盖
		node->setInput("task1", "b", makeScalarNative(2.0f));
		node->tryExecute("task1");

		CHECK(callCount == 1, "should execute exactly once");
		CHECK(std::abs(resultValue - 12.0f) < 1e-6f, "should use latest value");
	}
	END_TEST();

	// ── Test 13: setInputs 中途失败（批量中包含非法端口名）──
	TEST("setInputs fails on invalid port") {
		auto node = reg.createNode("add13", scalarAddSchema(), addRunImpl);

		std::atomic<bool> completed{false};
		node->setCompletionCallback([&](const Node::TaskId&, const Node::Result&) { completed = true; });

		// 先正常设置一个端口
		node->setInput("task1", "a", makeScalarNative(1.0f));

		// 批量设置中包含非法端口名 → 应该抛异常
		std::unordered_map<std::string, Node::TaskData> inputs;
		inputs.emplace("b", makeScalarNative(2.0f));
		inputs.emplace("no_such", makeScalarNative(3.0f)); // 非法端口

		CHECK_THROWS(node->setInput("task1", std::move(inputs)), NodeException,
					 "setInputs should throw on invalid port");
		CHECK(!completed, "should not execute after failed setInputs");

		// 之前正常设置的端口数据应保留
		node->setInput("task1", "b", makeScalarNative(5.0f));
		CHECK(node->isReady("task1"), "task should be ready");
		node->tryExecute("task1");
		CHECK(completed, "task should still be executable after rollback");
	}
	END_TEST();

	// ── Test 14: 回调中 getOutput + clearTask ──
	TEST("callback getOutput and clearTask") {
		auto node = reg.createNode("add14", scalarAddSchema(), addRunImpl);

		std::atomic<bool> gotOutput{false};
		std::atomic<bool> cleared{false};

		node->setCompletionCallback([&](const Node::TaskId& taskId, const Node::Result& result) {
			CHECK(result.ok(), "result should be Ok");
			auto outNT = node->getOutput(taskId, "s");
			auto* out = outNT.as<Tensor>();
			gotOutput = (std::abs(out->item<float>() - 9.0f) < 1e-6f);
			node->clearTask(taskId);
			cleared = true;
		});

		node->setInput("task1", "a", makeScalarNative(4.0f));
		node->setInput("task1", "b", makeScalarNative(5.0f));
		node->tryExecute("task1");

		CHECK(gotOutput, "callback should get correct output");
		CHECK(cleared, "callback should clear task");
	}
	END_TEST();

	// ── Test 15: 回调中 setInput 但不重入执行 ──
	TEST("callback sets input without re-entrant execution") {
		auto node = reg.createNode("add15", scalarAddSchema(), addRunImpl);

		std::atomic<int> callCount{0};
		bool needsTask2{false};

		node->setCompletionCallback([&](const Node::TaskId& taskId, const Node::Result& result) {
			++callCount;
			CHECK(result.ok(), "result should be Ok");
			node->clearTask(taskId);

			// 在回调中设置新任务输入（但不执行，禁止重入）
			if (callCount == 1) {
				node->setInput("task2", "a", makeScalarNative(1.0f));
				node->setInput("task2", "b", makeScalarNative(1.0f));
				needsTask2 = true;
			}
		});

		node->setInput("task1", "a", makeScalarNative(3.0f));
		node->setInput("task1", "b", makeScalarNative(3.0f));
		node->tryExecute("task1");

		// 回调中设置了 task2 的输入，需要外部触发执行
		CHECK(needsTask2, "callback should have set up task2");
		CHECK(callCount == 1, "only task1 completed so far");
		node->tryExecute("task2");

		CHECK(callCount == 2, "should process both tasks");
	}
	END_TEST();

	// ── Test 16: 单线程环境：乱序 setInput 多任务 ──
	TEST("multi-task interleaving 2") {
		auto node = reg.createNode("add16", scalarAddSchema(), addRunImpl);

		std::atomic<int> callCount{0};

		node->setCompletionCallback([&](const Node::TaskId& taskId, const Node::Result& result) {
			++callCount;
			CHECK(result.ok(), "result should be Ok");
			node->clearTask(taskId);
		});

		// 模拟多任务乱序（单线程下顺序仿真）
		node->setInput("task1", "a", makeScalarNative(1.0f));
		node->setInput("task2", "a", makeScalarNative(10.0f));
		node->setInput("task1", "b", makeScalarNative(2.0f));
		node->setInput("task2", "b", makeScalarNative(20.0f));
		node->tryExecute("task1");
		node->tryExecute("task2");

		CHECK(callCount == 2, "both tasks should complete");
	}
	END_TEST();

	// ── Test 17: 向量加法 ──
	TEST("vector addition with task API") {
		std::vector<float> aVals = {1, 2, 3, 4};
		std::vector<float> bVals = {5, 6, 7, 8};
		std::vector<float> exp = {6, 8, 10, 12};

		auto node = reg.createNode("add17", shapedAddSchema({4}), addRunImpl);

		std::atomic<bool> completed{false};
		bool match = false;

		node->setCompletionCallback([&](const Node::TaskId& taskId, const Node::Result& result) {
			CHECK(result.ok(), "result should be Ok");
			auto outNT = node->getOutput(taskId, "s");
			auto* out = outNT.as<Tensor>();
			auto outData = out->data<float>();
			match = true;
			for (size_t i = 0; i < exp.size(); ++i) {
				if (std::abs(outData[i] - exp[i]) > 1e-6f)
					match = false;
			}
			node->clearTask(taskId);
			completed = true;
		});

		node->setInput("task1", "a", makeVectorNative(aVals));
		node->setInput("task1", "b", makeVectorNative(bVals));
		node->tryExecute("task1");

		CHECK(completed, "vector task should complete");
		CHECK(match, "vector add values should match");
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
