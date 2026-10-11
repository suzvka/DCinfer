#include "SignalEvaluator.h"
#include "SignalStore.h"

#include <functional>
#include <unordered_map>
#include <utility>

namespace DC {

bool isSignalLogicType(const std::string& type) {
	return type == kSignalTypeSource || type == kSignalTypeConst || type == kSignalTypeNot ||
		   type == kSignalTypeAnd || type == kSignalTypeOr;
}

std::string signalWireName(const SignalNodePlan& producer, const std::string& port) {
	// 信号源：wire 即对外名，宿主 setSignal(name) 直达下游 gate，无需运行期搬运
	if (producer.isLogic && producer.type == kSignalTypeSource && !producer.signalName.empty())
		return producer.signalName;
	return producer.name + "." + port;
}

void evaluateStaticSignals(const SignalPlan& plan, SignalStore& store, const std::string& taskId) {
	if (plan.empty())
		return;

	std::unordered_map<std::string, const SignalNodePlan*> nodeByName;
	nodeByName.reserve(plan.nodes.size());
	for (const auto& n : plan.nodes)
		nodeByName[n.name] = &n;

	// 逻辑节点输入端子 → 上游 wire 名（来自信号边）；键 = dstNode\0dstPort
	std::unordered_map<std::string, std::string> inputWire;
	for (const auto& e : plan.edges) {
		std::string srcWire;
		auto it = nodeByName.find(e.srcNode);
		if (it != nodeByName.end())
			srcWire = signalWireName(*it->second, e.srcPort);
		else
			srcWire = e.srcNode + "." + e.srcPort;
		inputWire[e.dstNode + '\0' + e.dstPort] = std::move(srcWire);
	}

	// wire → 生产者 (node, port)
	std::unordered_map<std::string, std::pair<const SignalNodePlan*, std::string>> wireProducer;
	for (const auto& n : plan.nodes) {
		for (const auto& port : n.signalOutputs)
			wireProducer[signalWireName(n, port)] = {&n, port};
	}

	std::unordered_map<std::string, bool> memo;
	std::function<bool(const std::string&)> resolve = [&](const std::string& wire) -> bool {
		auto mit = memo.find(wire);
		if (mit != memo.end())
			return mit->second;
		// 环兜底：先占位 false，避免无限递归
		memo[wire] = false;

		bool val = false;
		auto pit = wireProducer.find(wire);
		if (pit == wireProducer.end()) {
			// 无生产者：外部/状态 wire，读 store 现值（宿主或运行期发布）
			val = store.get(wire, taskId, false);
		} else {
			const SignalNodePlan* prod = pit->second.first;
			if (prod->isLogic && prod->type == kSignalTypeSource) {
				val = store.get(prod->signalName.empty() ? wire : prod->signalName, taskId, false);
			} else if (prod->isLogic && prod->type == kSignalTypeConst) {
				val = prod->constValue;
			} else if (prod->isLogic && (prod->type == kSignalTypeNot || prod->type == kSignalTypeAnd ||
										 prod->type == kSignalTypeOr)) {
				// 收集已接线输入
				bool any = false;
				bool all = true;
				bool first = true;
				for (const auto& inPort : prod->signalInputs) {
					auto iit = inputWire.find(prod->name + '\0' + inPort);
					if (iit == inputWire.end())
						continue; // 未接线输入：跳过（AND 视为缺失不改变 all，OR 不贡献）
					bool b = resolve(iit->second);
					any = any || b;
					all = all && b;
					first = false;
				}
				if (prod->type == kSignalTypeNot) {
					// NOT 无接线输入时 fail-closed 取 false
					auto iit = inputWire.find(prod->name + '\0' + (prod->signalInputs.empty() ? std::string() : prod->signalInputs[0]));
					val = (iit == inputWire.end()) ? false : !resolve(iit->second);
				} else if (prod->type == kSignalTypeAnd) {
					val = !first && all;
				} else {
					val = any;
				}
			} else {
				// 计算节点状态输出：读 store（运行期发布；静态阶段缺省 false）
				val = store.get(wire, taskId, false);
			}
		}
		memo[wire] = val;
		return val;
	};

	// 写回逻辑节点输出 wire（task 级）；源由宿主驱动、常量在编译期已写全局，均跳过
	for (const auto& n : plan.nodes) {
		if (!n.isLogic)
			continue;
		if (n.type == kSignalTypeSource || n.type == kSignalTypeConst)
			continue;
		for (const auto& port : n.signalOutputs) {
			std::string wire = n.name + "." + port;
			store.set(wire, taskId, resolve(wire));
		}
	}
}

} // namespace DC
