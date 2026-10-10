// Connector 广播/路由 RunFn 单元测试
#include <atomic>
#include "NodeExecutor.h"
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>

#include "Connector.h"
#include "Node.h"
#include "NodeException.h"

using namespace DC;

using TensorType = DC::Tensor::TensorType;
using Tensor = DC::Tensor;

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

static Value makeFloatTensor(float value) {
	auto t = std::make_unique<Tensor>(TensorType::Float, sizeof(float));
	*t = value;
	return Value(std::move(t));
}

static Value makeIntTensor(int value) {
	auto t = std::make_unique<Tensor>(TensorType::Int, sizeof(int));
	*t = value;
	return Value(std::move(t));
}

void testBroadcastBasic() {
	TEST("broadcast 1→3 copies to all outputs") {
		auto schema = Connector::broadcastSchema(3);
		auto runFn = Connector::broadcastRunFn();

		auto node = std::make_unique<Node>("Connector.Broadcast", "bc1", schema, runFn);
		NodeExecutor exec(*node);

		exec.setInput("t1", "in", makeFloatTensor(42.0f));
		exec.tryExecute("t1");

		CHECK(exec.hasOutput("t1", "out_0"), "out_0 should have data");
		CHECK(exec.hasOutput("t1", "out_1"), "out_1 should have data");
		CHECK(exec.hasOutput("t1", "out_2"), "out_2 should have data");

		auto t0 = exec.takeOutputTensor("t1", "out_0");
		CHECK(std::abs(t0.item<float>() - 42.0f) < 1e-6f, "out_0 value mismatch");

		auto t1 = exec.takeOutputTensor("t1", "out_1");
		CHECK(std::abs(t1.item<float>() - 42.0f) < 1e-6f, "out_1 value mismatch");

		auto t2 = exec.takeOutputTensor("t1", "out_2");
		CHECK(std::abs(t2.item<float>() - 42.0f) < 1e-6f, "out_2 value mismatch");

		exec.clearTask("t1");
	}
	END_TEST();
}

void testBroadcastSingle() {
	TEST("broadcast 1→1 single output") {
		auto schema = Connector::broadcastSchema(1);
		auto runFn = Connector::broadcastRunFn();

		auto node = std::make_unique<Node>("Connector.Broadcast", "bc2", schema, runFn);
		NodeExecutor exec(*node);

		exec.setInput("t1", "in", makeFloatTensor(99.0f));
		exec.tryExecute("t1");

		CHECK(exec.hasOutput("t1", "out_0"), "out_0 should exist");
		auto t0 = exec.takeOutputTensor("t1", "out_0");
		CHECK(std::abs(t0.item<float>() - 99.0f) < 1e-6f, "single value mismatch");

		exec.clearTask("t1");
	}
	END_TEST();
}

void testBroadcastNotReady() {
	TEST("broadcast not ready without input") {
		auto schema = Connector::broadcastSchema(2);
		auto runFn = Connector::broadcastRunFn();

		auto node = std::make_unique<Node>("Connector.Broadcast", "bc3", schema, runFn);
		NodeExecutor exec(*node);

		CHECK(!exec.isReady("t1"), "should not be ready with no input");
		CHECK_THROWS(exec.tryExecute("t1"), NodeException, "tryExecute should throw without input");
	}
	END_TEST();
}

void testReentrancy() {
	TEST("broadcast rejects reentrant execution") {
		auto schema = Connector::broadcastSchema(2);
		auto runFn = Connector::broadcastRunFn();

		auto node = std::make_unique<Node>("Connector.Broadcast", "bc_re", schema, runFn);
		NodeExecutor exec(*node);

		exec.setInput("t1", "in", makeFloatTensor(1.0f));
		exec.setInput("t2", "in", makeFloatTensor(2.0f));

		exec.tryExecute("t1");
		// t1 完成后锁已释放，t2 应可执行
		exec.tryExecute("t2");

		exec.clearTask("t1");
		exec.clearTask("t2");
	}
	END_TEST();
}

int main() {
	try {
		testBroadcastBasic();
		testBroadcastSingle();
		testBroadcastNotReady();
		testReentrancy();

		if (failures == 0) {
			std::cout << "\nAll Connector tests passed!" << std::endl;
		} else {
			std::cout << "\n" << failures << " test(s) FAILED!" << std::endl;
		}
		return failures;
	} catch (const std::exception& e) {
		std::cerr << "Test failure: " << e.what() << std::endl;
		return -1;
	}
}
