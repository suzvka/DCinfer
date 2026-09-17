// Value 共享 / Tensor 冻结护栏 / 广播零拷贝共享 测试
//
// 覆盖值层共享契约：
//   - Value::share() 引用计数别名（零拷贝、指针等价）、move 保持独占
//   - Value::cloneOwned() 深拷贝产出独立可变副本；未注册类型明确报错
//   - Tensor::freeze() 预物化 + 写路径拒绝（TensorException::Frozen）、只读完好
//   - 冻结张量拷贝产出非冻结可变副本（clone 语义）
//   - Broadcast 语义：1:1 直通保持可变（零拷贝 move）；N>1 发布时冻结 + 共享
//   - 图集成：Broadcast(N) 扇出后所有下游收到同一 frozen payload（零拷贝共享）；
//     图级产出（takeOutput / 绑定输出）经发布标记产出独立可变副本
//   - 并发：冻结共享只读；未冻结共享首次物化（双重检查锁纵深保护）
#include <atomic>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "Connector.h"
#include "Node.h"
#include "NodeException.h"
#include "NodeExecutor.h"
#include "Tensor.hpp"
#include "TensorException.h"
#include "TestHarness.h"

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

// ── 辅助 ──

static Value makeFloatTensor(float value) {
	auto t = std::make_unique<Tensor>(TensorType::Float, sizeof(float));
	*t = value;
	return Value(std::move(t));
}

static Node::Schema identitySchema() {
	Node::Schema s;
	s.inputs = {{"x", Node::TensorType::Void, 0, {}}};
	s.outputs = {{"y", Node::TensorType::Void, 0, {}}};
	return s;
}

static Node::RunFn identityRunFn() {
	return [](Node::RunContext& ctx) -> Node::Result {
		if (!ctx.peek("x").as<Tensor>())
			return ctx.failure(Node::Status::InvalidInput, "identity: input is not a DC::Tensor");
		ctx.output("y", ctx.pop("x"));
		return ctx.success();
	};
}

// 捕获汇点：记录输入载荷指针 / 冻结态 / 值，并转发共享句柄到输出
struct ShareCapture {
	std::mutex mtx;
	std::vector<const Tensor*> payloads;
	std::vector<bool> frozenFlags;
	std::vector<float> values;
};

static Node::RunFn captureSinkRunFn(ShareCapture* cap) {
	return [cap](Node::RunContext& ctx) -> Node::Result {
		const auto& v = ctx.peek("x");
		const auto* t = v.as<Tensor>();
		if (!t)
			return ctx.failure(Node::Status::InvalidInput, "sink: input is not a DC::Tensor");
		auto span = t->data<float>();
		{
			std::lock_guard lk(cap->mtx);
			cap->payloads.push_back(t);
			cap->frozenFlags.push_back(t->isFrozen());
			cap->values.push_back(span.empty() ? -1.0f : span[0]);
		}
		ctx.output("y", ctx.pop("x")); // 转发共享句柄（保持共享链）
		return ctx.success();
	};
}

// 单输入多输出扇出节点（Broadcast RunFn 同款模式：1:1 直通 move；N>1 冻结 + 共享）
static Node::Schema fanOutSchema(size_t outputCount) {
	Node::Schema s;
	s.inputs = {{"in", Node::TensorType::Void, 0, {}}};
	s.outputs.reserve(outputCount);
	for (size_t i = 0; i < outputCount; ++i)
		s.outputs.push_back({"out_" + std::to_string(i), Node::TensorType::Void, 0, {}});
	return s;
}

static Node::RunFn fanOutRunFn() {
	return [](Node::RunContext& ctx) -> Node::Result {
		const auto& outputs = ctx.schema().outputs;
		if (outputs.size() == 1) {
			// 1:1 直通：零拷贝 move（保持可变，不冻结）
			ctx.output(outputs[0].name, ctx.pop("in"));
			return ctx.success();
		}
		// N>1：发布时一次性冻结 + 共享 N 份（零拷贝，只读共享）
		Value in = ctx.pop("in");
		if (auto* t = in.as<Tensor>())
			t->freeze();
		for (const auto& p : outputs)
			ctx.output(p.name, in.share());
		return ctx.success();
	};
}

// ── 1. Value 共享身份 ──

void testValueShareIdentity() {
	TEST("Value::share produces counted alias (zero-copy, same payload)") {
		auto t = std::make_unique<Tensor>(TensorType::Float, sizeof(float));
		*t = 3.5f;
		const void* raw = t.get();
		Value v(std::move(t));

		CHECK(!v.isShared(), "sole owner is not shared");
		CHECK(v.useCount() == 1, "use count 1 before aliasing");

		Value a = v.share();
		Value b = v.share();
		CHECK(v.isShared() && a.isShared() && b.isShared(), "shared after aliasing");
		CHECK(v.useCount() == 3, "use count 3 with two aliases");
		CHECK(a.get() == raw && b.get() == raw && v.get() == raw, "same payload pointer (zero-copy)");
		CHECK(a.as<Tensor>() == b.as<Tensor>(), "same typed payload");
		CHECK(std::abs(a.as<Tensor>()->item<float>() - 3.5f) < 1e-6f, "alias reads same value");
		CHECK(a.isPublished() && b.isPublished(), "aliases carry published flag");

		a = {};
		b = {};
		CHECK(!v.isShared() && v.useCount() == 1, "back to sole ownership after aliases released");
		// 粘性发布标记（#8-9）：别名消亡后源句柄仍视为已发布——载荷曾发布过，
		// 出口路径（takeOutput）据此产出独立副本，不误判独占交付
		CHECK(v.isPublished(), "source handle stays published after aliases die (sticky)");
	}
	END_TEST();
}

void testValueMoveSoleOwnership() {
	TEST("Value move keeps sole ownership (no refcount)") {
		Value a(std::make_unique<Tensor>(TensorType::Float, sizeof(float)));
		Value b(std::move(a));
		CHECK(b.useCount() == 1, "moved value remains sole owner");
		CHECK(!a, "moved-from value is empty");
		CHECK(!b.isShared(), "not shared after move");
	}
	END_TEST();
}

// ── 2. cloneOwned ──

void testCloneOwned() {
	TEST("cloneOwned: independent mutable copy; unregistered type errors") {
		auto t = std::make_unique<Tensor>(TensorType::Float, sizeof(float));
		*t = 6.0f;
		t->freeze();
		Value v(std::move(t));
		Value alias = v.share();

		Value c = v.cloneOwned();
		CHECK(c.get() != v.get(), "clone has separate payload");
		CHECK(c.useCount() == 1, "clone is sole-owned");
		auto* ct = c.as<Tensor>();
		CHECK(ct && !ct->isFrozen(), "clone is mutable (non-frozen)");
		*ct = 9.0f;
		CHECK(std::abs(ct->item<float>() - 9.0f) < 1e-6f, "clone value updated");
		CHECK(std::abs(v.as<Tensor>()->item<float>() - 6.0f) < 1e-6f, "original unaffected by clone mutation");

		// 未注册克隆函数（int 载荷）：明确报错
		Value vi(std::make_unique<int>(5));
		bool threw = false;
		try {
			(void)vi.cloneOwned();
		} catch (const NodeException&) {
			threw = true;
		}
		CHECK(threw, "unregistered type: cloneOwned throws NodeException");
	}
	END_TEST();
}

// ── 3. Tensor 冻结护栏 ──

void testTensorFreezeGuard() {
	TEST("freeze: cache pre-materialized, write paths rejected, reads intact") {
		Tensor t = Tensor::Create<float>({2});
		t[0].set<float>(3.0f);
		t[1].set<float>(4.0f);
		CHECK(!t.hasCache(), "lazy view mode before freeze");

		t.freeze();
		CHECK(t.isFrozen(), "frozen flag set");
		CHECK(t.hasCache(), "freeze pre-materializes dense cache (lazy path removed)");

		// 只读完好
		CHECK(std::abs(t[0].readScalar<float>() - 3.0f) < 1e-6f, "element read ok after freeze");
		auto span = t.data<float>();
		CHECK(span.size() == 2 && std::abs(span[1] - 4.0f) < 1e-6f, "dense read ok after freeze");

		// 写路径全部拒绝
		CHECK_THROWS(t[0].set<float>(1.0f), TensorException, "View::set rejected");
		CHECK_THROWS(t.fill<float>(1.0f), TensorException, "fill rejected");
		CHECK_THROWS(t = 1.0f, TensorException, "scalar assignment rejected");
		CHECK_THROWS(t.expand<float>({3}, 0.0f), TensorException, "expand rejected");
		CHECK_THROWS(t.crop({1}), TensorException, "crop rejected");
		CHECK_THROWS(t.loadData(Tensor::DataBlock(8), {2}), TensorException, "loadData rejected");
		CHECK_THROWS((void)t.getData<float>(), TensorException, "getData (consuming take) rejected");

		// 错误类型精确为 Frozen
		bool frozenErr = false;
		try {
			t[0].set<float>(1.0f);
		} catch (const TensorException& e) {
			frozenErr = (e.getErrorType() == TensorException::ErrorType::Frozen);
		}
		CHECK(frozenErr, "error type is Frozen");
	}
	END_TEST();
}

void testFrozenCopyMutable() {
	TEST("copy of frozen tensor yields mutable independent clone") {
		Tensor t = Tensor::Create<float>({2});
		t[0].set<float>(7.0f);
		t[1].set<float>(8.0f);
		t.freeze();

		Tensor c = t; // 拷贝构造
		CHECK(!c.isFrozen(), "copy is not frozen");
		c[0].set<float>(100.0f);
		CHECK(std::abs(c[0].readScalar<float>() - 100.0f) < 1e-6f, "copy mutated");
		CHECK(std::abs(t[0].readScalar<float>() - 7.0f) < 1e-6f, "source first element unchanged");
		CHECK(std::abs(t[1].readScalar<float>() - 8.0f) < 1e-6f, "source second element unchanged");
	}
	END_TEST();
}

// ── 4. 节点内扇出：1:1 直通 / N>1 冻结共享 ──

void testFanOutNodeSinglePassthrough() {
	TEST("fan-out node 1:1 passthrough stays mutable (zero-copy move)") {
		auto schema = fanOutSchema(1);
		auto node = std::make_unique<Node>("Builtin", "fan_one", schema, fanOutRunFn());
		NodeExecutor exec(*node);

		auto raw = std::make_unique<Tensor>(TensorType::Float, sizeof(float));
		*raw = 99.0f;
		Tensor* rawPtr = raw.get();
		exec.setInput("t1", "in", Value(std::move(raw)));
		exec.tryExecute("t1");

		Value out = exec.takeOutput("t1", "out_0");
		CHECK(out.as<Tensor>() == rawPtr, "1:1 passthrough: output is the same payload (zero-copy move)");
		CHECK(out.useCount() == 1, "sole ownership (no refcount introduced)");
		auto* t = out.as<Tensor>();
		CHECK(t && !t->isFrozen(), "1:1 passthrough is not frozen");
		*t = 5.0f; // 保持可变
		CHECK(std::abs(t->item<float>() - 5.0f) < 1e-6f, "mutation allowed on 1:1 output");

		exec.clearTask("t1");
	}
	END_TEST();
}

void testFanOutNodeIndependentTakes() {
	TEST("fan-out node 1:N: frozen share; takes are independent, siblings not corrupted") {
		auto schema = fanOutSchema(3);
		auto node = std::make_unique<Node>("Builtin", "fan_three", schema, fanOutRunFn());
		NodeExecutor exec(*node);

		exec.setInput("t1", "in", makeFloatTensor(42.0f));
		exec.tryExecute("t1");

		CHECK(exec.hasOutput("t1", "out_0"), "out_0 produced");
		CHECK(exec.hasOutput("t1", "out_1"), "out_1 produced");
		CHECK(exec.hasOutput("t1", "out_2"), "out_2 produced");

		// take 边界产出独立可变副本（冻结由克隆解除）
		auto t0 = exec.takeOutputTensor("t1", "out_0");
		CHECK(std::abs(t0.item<float>() - 42.0f) < 1e-6f, "out_0 value");
		CHECK(!t0.isFrozen(), "taken copy is mutable");
		t0 = 111.0f; // 突变副本

		// 兄弟 tap 不受影响（move-from-shared 回归）
		auto t1 = exec.takeOutputTensor("t1", "out_1");
		CHECK(std::abs(t1.item<float>() - 42.0f) < 1e-6f, "out_1 not corrupted by out_0 take/mutation");
		auto t2 = exec.takeOutputTensor("t1", "out_2");
		CHECK(std::abs(t2.item<float>() - 42.0f) < 1e-6f, "out_2 not corrupted");
		// 末份（兄弟份已全部消费）：发布标记兜底——仍产出独立可变副本
		CHECK(!t2.isFrozen(), "last taker also yields mutable copy (published flag)");
		t2 = 5.0f;
		CHECK(std::abs(t2.item<float>() - 5.0f) < 1e-6f, "last taken copy mutation allowed");

		exec.clearTask("t1");
	}
	END_TEST();
}

// ── 5. 图集成：Broadcast(N) 零拷贝共享 + 发布冻结 + 出口可变 ──

void testGraphFanOutSharedFrozen() {
	TEST("graph fan-out: Broadcast(N) sinks share one frozen payload (zero-copy)") {
		TestHarness harness;
		harness.addNode(std::make_unique<Node>("Builtin", "src", identitySchema(), identityRunFn()));

		ShareCapture cap;
		harness.addNode(std::make_unique<Node>("Builtin", "s0", identitySchema(), captureSinkRunFn(&cap)));
		harness.addNode(std::make_unique<Node>("Builtin", "s1", identitySchema(), captureSinkRunFn(&cap)));
		harness.addNode(std::make_unique<Node>("Builtin", "s2", identitySchema(), captureSinkRunFn(&cap)));

		// 显式广播连接器：src → bc → s0/s1/s2（1:N 分发姿势）
		auto bc = std::make_unique<Node>("Connector.Broadcast", "bc", Connector::broadcastSchema(3),
										 Connector::broadcastRunFn(), ThreadPoolAffinity::System);
		bc->setConnector(true);
		harness.addNode(std::move(bc));
		harness.connect("src", "y", "bc", "in");
		harness.connect("bc", "out_0", "s0", "x");
		harness.connect("bc", "out_1", "s1", "x");
		harness.connect("bc", "out_2", "s2", "x");

		harness.feedInput("t1", "src", "x", makeFloatTensor(21.0f));
		harness.submit("t1", {{"s0", "y"}, {"s1", "y"}, {"s2", "y"}});
		CHECK(harness.awaitCompletion("t1"), "task should complete");

		CHECK(cap.payloads.size() == 3, "all three sinks executed");
		if (cap.payloads.size() == 3) {
			CHECK(cap.payloads[0] == cap.payloads[1] && cap.payloads[1] == cap.payloads[2],
				  "fan-out: all sinks share the same payload (zero-copy)");
			CHECK(cap.frozenFlags[0] && cap.frozenFlags[1] && cap.frozenFlags[2],
				  "shared payload is frozen (read-only published)");
			CHECK(std::abs(cap.values[0] - 21.0f) < 1e-6f, "s0 value");
			CHECK(std::abs(cap.values[1] - 21.0f) < 1e-6f, "s1 value");
			CHECK(std::abs(cap.values[2] - 21.0f) < 1e-6f, "s2 value");
		}

		// 图级产出（声明输出经 takeOutput 出口捕获）：产出独立可变副本
		auto out = harness.getOutputTensor("t1", "s0", "y");
		CHECK(std::abs(out.item<float>() - 21.0f) < 1e-6f, "captured graph output value");
		CHECK(!out.isFrozen(), "captured graph output is mutable (independent clone)");
		out = 8.0f;
		CHECK(std::abs(out.item<float>() - 8.0f) < 1e-6f, "mutation on captured output allowed");
	}
	END_TEST();
}

void testBoundOutputTakeMutable() {
	TEST("bound output take (OutputZone path): published payload yields mutable tensor") {
		InferGraph g;
		g.addNode(std::make_unique<Node>("Builtin", "src", identitySchema(), identityRunFn()));
		g.addNode(std::make_unique<Node>("Builtin", "s0", identitySchema(), identityRunFn()));
		g.addNode(std::make_unique<Node>("Builtin", "s1", identitySchema(), identityRunFn()));
		auto bc = std::make_unique<Node>("Connector.Broadcast", "bc", Connector::broadcastSchema(2),
										 Connector::broadcastRunFn(), ThreadPoolAffinity::System);
		bc->setConnector(true);
		g.addNode(std::move(bc));
		g.connect("src", "y", "bc", "in");
		g.connect("bc", "out_0", "s0", "x");
		g.connect("bc", "out_1", "s1", "x");
		g.bindOutput("o0", "s0", "y");

		g.feedInput("t1", "src", "x", makeFloatTensor(4.5f));
		g.submitBound("t1");
		CHECK(g.waitForResult("t1").status == TaskStatus::Succeeded, "task should succeed");

		// 绑定输出经 OutputZone 出口：发布残留载荷产出独立可变副本
		auto out = g.takeOutputTensor("t1", "s0", "y");
		CHECK(std::abs(out.item<float>() - 4.5f) < 1e-6f, "bound output value");
		CHECK(!out.isFrozen(), "bound output take is mutable (independent clone)");
		out = 2.5f;
		CHECK(std::abs(out.item<float>() - 2.5f) < 1e-6f, "mutation on taken tensor allowed");
	}
	END_TEST();
}

// ── 6. 并发 ──

void testConcurrentReadFrozenShared() {
	TEST("concurrent reads of frozen shared payload are consistent") {
		auto t = std::make_unique<Tensor>(TensorType::Float, sizeof(float), Tensor::Shape{4});
		t->fill<float>(1.5f);
		t->freeze();
		CHECK(t->hasCache(), "frozen payload has pre-materialized cache");
		Value base(std::move(t));

		std::atomic<int> bad{0};
		std::vector<std::thread> threads;
		for (int k = 0; k < 4; ++k) {
			threads.emplace_back([alias = base.share(), &bad]() {
				auto* tt = alias.as<Tensor>();
				for (int iter = 0; iter < 200; ++iter) {
					auto span = tt->data<float>();
					float sum = 0.0f;
					for (float x : span)
						sum += x;
					if (std::abs(sum - 6.0f) > 1e-3f)
						++bad;
				}
			});
		}
		for (auto& th : threads)
			th.join();

		CHECK(bad.load() == 0, "all concurrent reads saw consistent data");
		CHECK(base.useCount() == 1, "aliases released after join");
	}
	END_TEST();
}

void testConcurrentMaterializationLazy() {
	TEST("concurrent first read of shared lazy payload materializes once (DCL)") {
		auto t = std::make_unique<Tensor>(TensorType::Float, sizeof(float), Tensor::Shape{8});
		for (int i = 0; i < 8; ++i)
			(*t)[i].set<float>(2.0f);
		CHECK(!t->hasCache(), "view mode before first read");
		Value base(std::move(t));

		std::atomic<int> bad{0};
		std::vector<std::thread> threads;
		for (int k = 0; k < 4; ++k) {
			threads.emplace_back([alias = base.share(), &bad]() {
				auto* tt = alias.as<Tensor>();
				auto span = tt->data<float>(); // 并发首次物化
				float sum = 0.0f;
				for (float x : span)
					sum += x;
				if (std::abs(sum - 16.0f) > 1e-3f)
					++bad;
			});
		}
		for (auto& th : threads)
			th.join();

		CHECK(bad.load() == 0, "concurrent first reads consistent");
		CHECK(base.as<Tensor>()->hasCache(), "cache materialized");
	}
	END_TEST();
}

int main() {
	try {
		testValueShareIdentity();
		testValueMoveSoleOwnership();
		testCloneOwned();

		testTensorFreezeGuard();
		testFrozenCopyMutable();

		testFanOutNodeSinglePassthrough();
		testFanOutNodeIndependentTakes();

		testGraphFanOutSharedFrozen();
		testBoundOutputTakeMutable();

		testConcurrentReadFrozenShared();
		testConcurrentMaterializationLazy();

		if (failures == 0) {
			std::cout << "\nAll ValueSharing tests passed!" << std::endl;
		} else {
			std::cout << "\n" << failures << " test(s) FAILED!" << std::endl;
		}
		return failures;
	} catch (const std::exception& e) {
		std::cerr << "Test failure: " << e.what() << std::endl;
		return -1;
	}
}
