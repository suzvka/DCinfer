// SignalEvaluator 单元测试：静态信号面求值（source/const/AND/OR/NOT → wire）
#include "SignalEvaluator.h"
#include "SignalPlan.h"
#include "SignalStore.h"

#include <iostream>
#include <stdexcept>
#include <string>

using namespace DC;

namespace {

SignalNodePlan source(const std::string& name, const std::string& signalName) {
	SignalNodePlan n;
	n.name = name;
	n.type = kSignalTypeSource;
	n.isLogic = true;
	n.signalName = signalName;
	n.signalOutputs = {kSigOut};
	return n;
}

SignalNodePlan konst(const std::string& name, bool value) {
	SignalNodePlan n;
	n.name = name;
	n.type = kSignalTypeConst;
	n.isLogic = true;
	n.hasConstValue = true;
	n.constValue = value;
	n.signalOutputs = {kSigOut};
	return n;
}

SignalNodePlan logic(const std::string& name, const std::string& type, std::vector<std::string> inputs) {
	SignalNodePlan n;
	n.name = name;
	n.type = type;
	n.isLogic = true;
	n.signalInputs = std::move(inputs);
	n.signalOutputs = {kSigOut};
	return n;
}

SignalEdgePlan edge(const std::string& sn, const std::string& dn, const std::string& dp) {
	return SignalEdgePlan{sn, kSigOut, dn, dp};
}

void expect(bool cond, const char* msg) {
	if (!cond)
		throw std::runtime_error(msg);
}

} // namespace

static void runTests() {
	const std::string task = "t1";

	// AND：两源全 true 才 true
	{
		SignalPlan plan;
		plan.nodes = {source("s1", "a"), source("s2", "b"), logic("and1", kSignalTypeAnd, {"sig.in0", "sig.in1"})};
		plan.edges = {edge("s1", "and1", "sig.in0"), edge("s2", "and1", "sig.in1")};
		SignalStore store;
		store.set("a", true);
		store.set("b", true);
		evaluateStaticSignals(plan, store, task);
		expect(store.get("and1.sig.out", task, false) == true, "AND(true,true) must be true");
		store.set("b", false);
		evaluateStaticSignals(plan, store, task);
		expect(store.get("and1.sig.out", task, false) == false, "AND(true,false) must be false");
	}
	std::cout << "Test passed: AND over two sources" << std::endl;

	// OR：任一 true 即 true
	{
		SignalPlan plan;
		plan.nodes = {source("s1", "a"), source("s2", "b"), logic("or1", kSignalTypeOr, {"sig.in0", "sig.in1"})};
		plan.edges = {edge("s1", "or1", "sig.in0"), edge("s2", "or1", "sig.in1")};
		SignalStore store;
		store.set("a", false);
		store.set("b", true);
		evaluateStaticSignals(plan, store, task);
		expect(store.get("or1.sig.out", task, false) == true, "OR(false,true) must be true");
	}
	std::cout << "Test passed: OR over two sources" << std::endl;

	// NOT + 常量：NOT(const true) = false
	{
		SignalPlan plan;
		plan.nodes = {konst("c", true), logic("not1", kSignalTypeNot, {"sig.in0"})};
		plan.edges = {edge("c", "not1", "sig.in0")};
		SignalStore store;
		evaluateStaticSignals(plan, store, task);
		expect(store.get("not1.sig.out", task, false) == false, "NOT(const true) must be false");
	}
	std::cout << "Test passed: NOT over const" << std::endl;

	// 级联：AND(s1,s2) → NOT；task 级隔离
	{
		SignalPlan plan;
		plan.nodes = {source("s1", "a"), source("s2", "b"), logic("and1", kSignalTypeAnd, {"sig.in0", "sig.in1"}),
					  logic("not1", kSignalTypeNot, {"sig.in0"})};
		plan.edges = {edge("s1", "and1", "sig.in0"), edge("s2", "and1", "sig.in1"), edge("and1", "not1", "sig.in0")};
		SignalStore store;
		store.set("a", true);
		store.set("b", true);
		evaluateStaticSignals(plan, store, task);
		expect(store.get("not1.sig.out", task, false) == false, "NOT(AND(true,true)) must be false");
		// task 级覆盖：t1 视 a=false，全局仍 true
		store.set("a", task, false);
		evaluateStaticSignals(plan, store, task);
		expect(store.get("and1.sig.out", task, false) == false, "task-level a=false must drive AND false");
		expect(store.get("not1.sig.out", task, false) == true, "NOT(AND false) must be true");
	}
	std::cout << "Test passed: cascaded AND→NOT with task-level override" << std::endl;

	std::cout << "\nAll SignalEvaluator tests passed!" << std::endl;
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
