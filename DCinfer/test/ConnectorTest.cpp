// Connector 广播/路由 RunFn 单元测试
#include <atomic>
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>

#include "Connector.h"
#include "Node.h"

using namespace DC;

using TensorType = DC::Tensor::TensorType;
using Tensor     = DC::Tensor;

static int failures = 0;

#define CHECK(cond, msg) do { \
	if (!(cond)) { \
		std::cerr << "FAIL: " << msg << std::endl; \
		++failures; \
		return; \
	} \
} while(0)

#define TEST(name) std::cout << "Test: " << name << " ... " << std::flush; \
	[&]()
#define END_TEST() (); \
	std::cout << "PASSED" << std::endl

// ── 辅助函数 ──

static Value makeFloatTensor(float value) {
	auto* p = new Tensor(TensorType::Float, sizeof(float));
	*p = value;
	return Value(p, [](Tensor* ptr) { delete ptr; });
}

static Value makeIntTensor(int value) {
	auto* p = new Tensor(TensorType::Int, sizeof(int));
	*p = value;
	return Value(p, [](Tensor* ptr) { delete ptr; });
}

// ── 广播连接器测试 ──

void testBroadcastBasic() {
	TEST("broadcast 1→3 copies to all outputs") {
		auto schema = Connector::broadcastSchema(3);
		auto runFn  = Connector::broadcastRunFn();

		auto node = std::make_unique<Node>("Connector.Broadcast", "bc1", schema, runFn);

		node->setInput("t1", "in", makeFloatTensor(42.0f));
		CHECK(node->tryExecute("t1"), "broadcast tryExecute should succeed");

		// 验证所有三个输出口都有数据
		CHECK(node->hasOutput("t1", "out_0"), "out_0 should have data");
		CHECK(node->hasOutput("t1", "out_1"), "out_1 should have data");
		CHECK(node->hasOutput("t1", "out_2"), "out_2 should have data");

		// 验证数据正确性
		auto t0 = node->getOutputTensor("t1", "out_0");
		CHECK(std::abs(t0.item<float>() - 42.0f) < 1e-6f, "out_0 value mismatch");

		auto t1 = node->getOutputTensor("t1", "out_1");
		CHECK(std::abs(t1.item<float>() - 42.0f) < 1e-6f, "out_1 value mismatch");

		auto t2 = node->getOutputTensor("t1", "out_2");
		CHECK(std::abs(t2.item<float>() - 42.0f) < 1e-6f, "out_2 value mismatch");

		node->clearTask("t1");
	}
	END_TEST();
}

void testBroadcastSingle() {
	TEST("broadcast 1→1 single output") {
		auto schema = Connector::broadcastSchema(1);
		auto runFn  = Connector::broadcastRunFn();

		auto node = std::make_unique<Node>("Connector.Broadcast", "bc2", schema, runFn);

		node->setInput("t1", "in", makeFloatTensor(99.0f));
		CHECK(node->tryExecute("t1"), "single broadcast tryExecute should succeed");

		CHECK(node->hasOutput("t1", "out_0"), "out_0 should exist");
		auto t0 = node->getOutputTensor("t1", "out_0");
		CHECK(std::abs(t0.item<float>() - 99.0f) < 1e-6f, "single value mismatch");

		node->clearTask("t1");
	}
	END_TEST();
}

void testBroadcastNotReady() {
	TEST("broadcast not ready without input") {
		auto schema = Connector::broadcastSchema(2);
		auto runFn  = Connector::broadcastRunFn();

		auto node = std::make_unique<Node>("Connector.Broadcast", "bc3", schema, runFn);

		CHECK(!node->isReady("t1"), "should not be ready with no input");
		CHECK(!node->tryExecute("t1"), "tryExecute should fail without input");
	}
	END_TEST();
}

// ── 路由连接器测试 ──

void testRoutingRoundRobin() {
	TEST("routing 1→3 round-robin distribution") {
		auto schema = Connector::routingSchema(3);
		auto runFn  = Connector::routingRunFn();

		auto node = std::make_unique<Node>("Connector.Routing", "rt1", schema, runFn);

		// task1 → out_0
		node->setInput("t1", "in", makeFloatTensor(1.0f));
		CHECK(node->tryExecute("t1"), "routing t1 execute");
		CHECK(node->hasOutput("t1", "out_0"), "t1 should go to out_0");
		CHECK(!node->hasOutput("t1", "out_1"), "t1 should NOT go to out_1");
		CHECK(!node->hasOutput("t1", "out_2"), "t1 should NOT go to out_2");
		{
			auto t = node->getOutputTensor("t1", "out_0");
			CHECK(std::abs(t.item<float>() - 1.0f) < 1e-6f, "t1 value");
		}
		node->clearTask("t1");

		// task2 → out_1
		node->setInput("t2", "in", makeFloatTensor(2.0f));
		CHECK(node->tryExecute("t2"), "routing t2 execute");
		CHECK(node->hasOutput("t2", "out_1"), "t2 should go to out_1");
		CHECK(!node->hasOutput("t2", "out_0"), "t2 should NOT go to out_0");
		CHECK(!node->hasOutput("t2", "out_2"), "t2 should NOT go to out_2");
		node->clearTask("t2");

		// task3 → out_2
		node->setInput("t3", "in", makeFloatTensor(3.0f));
		CHECK(node->tryExecute("t3"), "routing t3 execute");
		CHECK(node->hasOutput("t3", "out_2"), "t3 should go to out_2");
		node->clearTask("t3");

		// task4 → out_0 (wrap around)
		node->setInput("t4", "in", makeFloatTensor(4.0f));
		CHECK(node->tryExecute("t4"), "routing t4 execute");
		CHECK(node->hasOutput("t4", "out_0"), "t4 should wrap to out_0");
		node->clearTask("t4");
	}
	END_TEST();
}

void testRoutingSingleOutput() {
	TEST("routing 1→1 always hits out_0") {
		auto schema = Connector::routingSchema(1);
		auto runFn  = Connector::routingRunFn();

		auto node = std::make_unique<Node>("Connector.Routing", "rt2", schema, runFn);

		for (int i = 0; i < 5; ++i) {
			auto tid = "t" + std::to_string(i);
			node->setInput(tid, "in", makeIntTensor(i));
			CHECK(node->tryExecute(tid), "routing single execute");
			CHECK(node->hasOutput(tid, "out_0"), "should always hit out_0");
			auto t = node->getOutputTensor(tid, "out_0");
			CHECK(t.item<int>() == i, "value mismatch");
			node->clearTask(tid);
		}
	}
	END_TEST();
}

// ── 重入保护测试 ──

void testReentrancy() {
	TEST("broadcast rejects reentrant execution") {
		auto schema = Connector::broadcastSchema(2);
		auto runFn  = Connector::broadcastRunFn();

		auto node = std::make_unique<Node>("Connector.Broadcast", "bc_re", schema, runFn);

		// 注入两个不同 task 的数据
		node->setInput("t1", "in", makeFloatTensor(1.0f));
		node->setInput("t2", "in", makeFloatTensor(2.0f));

		// 第一个执行成功
		CHECK(node->tryExecute("t1"), "t1 should execute");
		// 第二个也应该能执行（t1 已完成，锁已释放）
		CHECK(node->tryExecute("t2"), "t2 should execute after t1");

		node->clearTask("t1");
		node->clearTask("t2");
	}
	END_TEST();
}

int main() {
	try {
		testBroadcastBasic();
		testBroadcastSingle();
		testBroadcastNotReady();
		testRoutingRoundRobin();
		testRoutingSingleOutput();
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
