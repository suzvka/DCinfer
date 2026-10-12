#include "Ir/GraphCompiler.h"

#include "Connector.h"
#include "EngineRegistry.h"
#include "GraphException.h"
#include "SignalEvaluator.h"
#include "Ir/DcgArchive.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>

namespace DC::Ir {

namespace {

/// JSON 输入总字节上限：入口拒绝异常图定义，防 nlohmann 解析内存放大；
/// .dcg 路径解压侧已有预算，此处补齐 graph.json 字符串层防线。
constexpr std::size_t kMaxGraphJsonBytes = 32ull * 1024 * 1024;
/// 节点数上限；buildGraph 为全部编译路径的必经点。
constexpr std::size_t kMaxGraphNodes = 4096;

} // namespace

// Node::Port::shape 为 vector<int64_t>：-1 动态维直通即可保证 roundtrip 稳定。
// TensorData::Shape 为 vector<size_t>，不能表达 -1：隐式转换翻成 size_t::max
// 且形状乘积溢出。属核心库数据路径的已知限制，见 GraphCompiler.h，不在 DCIr 侧解决。

static std::string affinityToString(ResourceClass a) {
	switch (a) {
	case ResourceClass::Compute: return "Compute";
	case ResourceClass::Operator: return "Operator";
	case ResourceClass::System: return "System";
	}
	return "Operator";
}

static ResourceClass stringToAffinity(const std::string& s) {
	if (s == "Compute") return ResourceClass::Compute;
	if (s == "Operator") return ResourceClass::Operator;
	if (s == "System") return ResourceClass::System;
	return ResourceClass::Operator;
}

nlohmann::json GraphCompiler::portToJson(const Node::Port& port) {
	nlohmann::json j;
	j["name"] = port.name;
	j["tensorType"] = TensorMeta::typeToString(port.type);
	j["typeSize"] = static_cast<int64_t>(port.typeSize);
	nlohmann::json shapeArr = nlohmann::json::array();
	for (auto dim : port.shape) {
		shapeArr.push_back(dim);
	}
	j["shape"] = std::move(shapeArr);
	j["required"] = port.required;
	return j;
}

static Node::Port jsonToPort(const nlohmann::json& j) {
	Node::Port p;
	p.name = j.at("name").get<std::string>();
	p.type = TensorMeta::stringToType(j.at("tensorType").get<std::string>());
	// 负值经 static_cast<size_t> 会穿透为 SIZE_MAX，显式拒绝；0 合法，即 Void 加 0。
	const int64_t typeSize = j.at("typeSize").get<int64_t>();
	if (typeSize < 0 || typeSize > (1ll << 20)) {
		throw GraphException(GraphException::ErrorType::Other, "GraphCompiler::jsonToPort",
			"invalid typeSize " + std::to_string(typeSize) + " for port '" + p.name + "'");
	}
	p.typeSize = static_cast<size_t>(typeSize);
	for (auto& dim : j.at("shape")) {
		// 禁止有符号与无符号转换：-1 会被转为巨大值，破坏 roundtrip 对称性。
		p.shape.push_back(dim.get<int64_t>());
	}
	p.required = j.value("required", true);
	return p;
}

/// @brief JSON 端口是否信号端子（kind == "signal"）；缺省视为张量端口。
static bool isSignalPortJson(const nlohmann::json& p) {
	return p.is_object() && p.value("kind", std::string("tensor")) == "signal";
}

/// @brief 信号端子序列化（仅 name/kind/required；无张量形状）。
static nlohmann::json signalPortToJson(const std::string& name) {
	nlohmann::json j;
	j["name"] = name;
	j["kind"] = "signal";
	j["required"] = false;
	return j;
}

/// @brief 将 JSON 中的 tag / modelPath 应用到已创建节点，三分支共用。
///        modelPath 为引擎自定义不透明信息，原样透传，不拼接、不校验、不做文件系统解释。
///        控制信号改由信号面（sig.gate + 信号边）降级，见 buildGraph；不再走节点级 signal 名。
static void applyNodeMeta(DC::Node& node, const nlohmann::json& j) {
	if (j.contains("modelPath")) {
		node.setModelPath(j["modelPath"].get<std::string>());
	}
	if (j.contains("tag")) {
		node.setTag(j["tag"].get<std::string>());
	}
}

nlohmann::json GraphCompiler::edgesToJson(const InferGraph& graph) {
	nlohmann::json edgesArr = nlohmann::json::array();

	// 索引：connector 名映射到其所有输出边。edges 返回值为副本，快照必须
	// 钉在局部变量上延长生命周期，否则 range-for 后元素指针悬垂。
	const auto edgesSnapshot = graph.edges();
	std::map<std::string, std::vector<const InferGraph::Edge*>, std::less<>> connectorOut;
	for (const auto& e : edgesSnapshot) {
		auto* srcNode = graph.node(e.srcNode);
		if (srcNode && srcNode->isConnector()) {
			connectorOut[e.srcNode].push_back(&e);
		}
	}

	auto isFanOutConnector = [&](const std::string& name) {
		auto* n = graph.node(name);
		if (!n || !n->isConnector() || n->type().find("Broadcast") == std::string::npos) return false;
		auto it = connectorOut.find(name);
		return it != connectorOut.end() && it->second.size() > 1;
	};

	// 遍历 processor 源边：指向连接器时穿透链，展开为 processor 到 processor 逻辑边；
	// 链上有分发 Broadcast 则标 mode=broadcast 以重建时还原分发组，1:1 链折叠为直连边。
	for (auto& e : graph.edges()) {
		auto* srcNode = graph.node(e.srcNode);
		if (!srcNode || srcNode->isConnector()) continue;
		auto* dstNode = graph.node(e.dstNode);
		if (!dstNode) continue;

		if (!dstNode->isConnector()) {
			nlohmann::json edge;
			edge["srcNode"] = e.srcNode;
			edge["srcPort"] = e.srcPort;
			edge["dstNode"] = e.dstNode;
			edge["dstPort"] = e.dstPort;
			edgesArr.push_back(std::move(edge));
			continue;
		}

		// 连接器链穿透：迭代收集所有逻辑终点，即非连接器节点
		struct WalkItem {
			std::string nodeName;
			std::string portName;
			bool fanOutPath;
		};
		std::vector<WalkItem> stack;
		stack.push_back({e.dstNode, e.dstPort, isFanOutConnector(e.dstNode)});
		std::set<std::pair<std::string, std::string>> visited; // (节点, 入口) 防环

		while (!stack.empty()) {
			WalkItem item = std::move(stack.back());
			stack.pop_back();
			if (!visited.insert({item.nodeName, item.portName}).second) continue;

			auto* cur = graph.node(item.nodeName);
			if (!cur) continue;

			if (!cur->isConnector()) {
				nlohmann::json edge;
				edge["srcNode"] = e.srcNode;
				edge["srcPort"] = e.srcPort;
				edge["dstNode"] = item.nodeName;
				edge["dstPort"] = item.portName;
				if (item.fanOutPath) {
					edge["mode"] = "broadcast";
				}
				edgesArr.push_back(std::move(edge));
				continue;
			}

			auto it = connectorOut.find(item.nodeName);
			if (it == connectorOut.end()) continue;
			for (auto* outE : it->second) {
				stack.push_back({outE->dstNode, outE->dstPort,
								 item.fanOutPath || isFanOutConnector(outE->srcNode)});
			}
		}
	}
	return edgesArr;
}

nlohmann::json GraphCompiler::graphToJson(const InferGraph& graph) {
	nlohmann::json root;
	root["version"] = "1.0";

	const auto& plan = graph.signalPlan();
	std::map<std::string, const SignalNodePlan*> planByName;
	for (const auto& n : plan.nodes)
		planByName[n.name] = &n;

	// 节点：只序列化非连接器节点（处理器）；张量端口来自 schema，信号端子来自 plan
	nlohmann::json nodesArr = nlohmann::json::array();
	for (auto& name : graph.nodeNames()) {
		auto* node = graph.node(name);
		if (!node || node->isConnector()) continue;

		nlohmann::json j;
		j["name"] = node->name();
		j["type"] = node->type();
		if (!node->modelPath().empty()) {
			j["modelPath"] = node->modelPath();
		}
		j["affinity"] = affinityToString(node->affinity());
		if (!node->tag().empty()) {
			j["tag"] = node->tag();
		}

		auto pit = planByName.find(node->name());
		const SignalNodePlan* np = (pit != planByName.end()) ? pit->second : nullptr;

		nlohmann::json inputs = nlohmann::json::array();
		for (auto& port : node->schema().inputs) {
			inputs.push_back(portToJson(port));
		}
		if (np) {
			for (auto& sp : np->signalInputs)
				inputs.push_back(signalPortToJson(sp));
		}
		j["inputs"] = std::move(inputs);

		nlohmann::json outputs = nlohmann::json::array();
		for (auto& port : node->schema().outputs) {
			outputs.push_back(portToJson(port));
		}
		if (np) {
			for (auto& sp : np->signalOutputs)
				outputs.push_back(signalPortToJson(sp));
		}
		j["outputs"] = std::move(outputs);

		nodesArr.push_back(std::move(j));
	}

	// 逻辑节点：纯信号面，未物化为 DC::Node，从 plan 直接回写
	for (const auto& n : plan.nodes) {
		if (!n.isLogic) continue;
		nlohmann::json j;
		j["name"] = n.name;
		j["type"] = n.type;
		j["affinity"] = "System";
		if (n.type == kSignalTypeSource)
			j["signalName"] = n.signalName;
		if (n.type == kSignalTypeConst && n.hasConstValue)
			j["signalValue"] = n.constValue;
		nlohmann::json inputs = nlohmann::json::array();
		for (auto& sp : n.signalInputs)
			inputs.push_back(signalPortToJson(sp));
		j["inputs"] = std::move(inputs);
		nlohmann::json outputs = nlohmann::json::array();
		for (auto& sp : n.signalOutputs)
			outputs.push_back(signalPortToJson(sp));
		j["outputs"] = std::move(outputs);
		nodesArr.push_back(std::move(j));
	}
	root["nodes"] = std::move(nodesArr);

	// 边：张量边（穿透连接器）+ 信号边（原样回写）
	nlohmann::json edgesArr = edgesToJson(graph);
	for (const auto& e : plan.edges) {
		nlohmann::json ej;
		ej["srcNode"] = e.srcNode;
		ej["srcPort"] = e.srcPort;
		ej["dstNode"] = e.dstNode;
		ej["dstPort"] = e.dstPort;
		edgesArr.push_back(std::move(ej));
	}
	root["edges"] = std::move(edgesArr);

	// 输出绑定：alias 为签名元数据，不参与运行时寻址
	nlohmann::json bindingsArr = nlohmann::json::array();
	for (auto& b : graph.outputBindings()) {
		auto* boundNode = graph.node(b.nodeName);
		if (boundNode && boundNode->isConnector()) continue;
		nlohmann::json jb;
		jb["alias"] = b.alias;
		jb["nodeName"] = b.nodeName;
		jb["portName"] = b.portName;
		bindingsArr.push_back(std::move(jb));
	}
	root["outputBindings"] = std::move(bindingsArr);

	// 输入绑定：alias 为签名元数据，不参与运行时寻址
	nlohmann::json inputBindingsArr = nlohmann::json::array();
	for (auto& b : graph.inputBindings()) {
		auto* boundNode = graph.node(b.nodeName);
		if (boundNode && boundNode->isConnector()) continue;
		nlohmann::json jb;
		jb["alias"] = b.alias;
		jb["nodeName"] = b.nodeName;
		jb["portName"] = b.portName;
		inputBindingsArr.push_back(std::move(jb));
	}
	root["inputBindings"] = std::move(inputBindingsArr);

	return root;
}

void GraphCompiler::rebuildEdges(InferGraph& graph, const nlohmann::json& edgesJson) {
	if (!edgesJson.is_array()) return;

	struct EdgeTarget {
		std::string dstNode;
		std::string dstPort;
	};

	struct EdgeKey {
		std::string srcNode;
		std::string srcPort;
		std::string mode;
		bool operator<(const EdgeKey& o) const {
			if (srcNode != o.srcNode) return srcNode < o.srcNode;
			if (srcPort != o.srcPort) return srcPort < o.srcPort;
			return mode < o.mode;
		}
	};

	std::map<EdgeKey, std::vector<EdgeTarget>> groups;

	for (auto& e : edgesJson) {
		EdgeKey key;
		key.srcNode = e.at("srcNode").get<std::string>();
		key.srcPort = e.at("srcPort").get<std::string>();
		key.mode = e.value("mode", "");

		EdgeTarget tgt;
		tgt.dstNode = e.at("dstNode").get<std::string>();
		tgt.dstPort = e.at("dstPort").get<std::string>();
		groups[std::move(key)].push_back(std::move(tgt));
	}

	size_t connId = 0;
	for (auto& [key, targets] : groups) {
		if (targets.empty()) continue;

		if (key.mode == "routing") {
			// Routing 已移除，轮询属业务语义：旧图显式报错而非静默降级。
			throw DC::GraphException(DC::GraphException::ErrorType::Other,
									 "GraphCompiler::rebuildEdges",
									 "edge mode \"routing\" is no longer supported; express "
									 "round-robin via a custom node with per-instance selection");
		}

		if (key.mode == "broadcast") {
			size_t n = targets.size();
			Node::Schema connSchema;
			Node::RunFn connRunFn;
			std::string connType;
			connSchema = DC::Connector::broadcastSchema(n);
			connRunFn = DC::Connector::broadcastRunFn();
			connType = "Connector.Broadcast";
			std::string connName = "__" + key.mode + "_" + std::to_string(connId++);

			auto connNode = std::make_unique<DC::Node>(
				connType, connName, std::move(connSchema), std::move(connRunFn),
				ResourceClass::System);
			connNode->setConnector(true);
			graph.addNode(std::move(connNode));

			// connect 自动包裹直通导线，序列化折叠后不可见且 round-trip 幂等；
			// 连接失败直接透传 GraphException，反序列化 fail-fast。
			graph.connect(key.srcNode, key.srcPort, connName, "in");
			for (size_t i = 0; i < targets.size(); ++i) {
				graph.connect(connName, "out_" + std::to_string(i), targets[i].dstNode, targets[i].dstPort);
			}
		} else {
			// 默认 1 对 1：connect 自动插入导线连接器；失败 fail-fast，同上。
			for (auto& tgt : targets) {
				graph.connect(key.srcNode, key.srcPort, tgt.dstNode, tgt.dstPort);
			}
		}
	}
}

void GraphCompiler::buildGraph(InferGraph& graph, const nlohmann::json& root) {

	// 节点数预算：json 字符串 / .dcg 两条路径都经此
	if (root.contains("nodes") && root["nodes"].is_array()
		&& root["nodes"].size() > kMaxGraphNodes) {
		throw GraphException(GraphException::ErrorType::Other, "GraphCompiler::buildGraph",
							 "too many nodes in graph definition (" + std::to_string(root["nodes"].size())
								 + " > limit " + std::to_string(kMaxGraphNodes) + ")");
	}

	static const nlohmann::json kEmptyArr = nlohmann::json::array();
	SignalPlan plan;
	auto& reg = EngineRegistry::instance();

	// 第一遍：建节点。张量端口入 schema；信号端子剔离入 plan；逻辑节点不物化为 DC::Node。
	for (auto& j : root.at("nodes")) {
		std::string name = j.at("name").get<std::string>();
		std::string type = j.at("type").get<std::string>();

		Node::Schema schema;
		SignalNodePlan np;
		np.name = name;
		const bool isLogic = isSignalLogicType(type);
		if (isLogic) {
			np.isLogic = true;
			np.type = type;
			if (j.contains("signalName") && j["signalName"].is_string())
				np.signalName = j["signalName"].get<std::string>();
			if (j.contains("signalValue") && j["signalValue"].is_boolean()) {
				np.hasConstValue = true;
				np.constValue = j["signalValue"].get<bool>();
			}
		}
		const nlohmann::json& inArr = (j.contains("inputs") && j["inputs"].is_array()) ? j["inputs"] : kEmptyArr;
		for (auto& p : inArr) {
			if (isSignalPortJson(p)) np.signalInputs.push_back(p.at("name").get<std::string>());
			else schema.inputs.push_back(jsonToPort(p));
		}
		const nlohmann::json& outArr = (j.contains("outputs") && j["outputs"].is_array()) ? j["outputs"] : kEmptyArr;
		for (auto& p : outArr) {
			if (isSignalPortJson(p)) np.signalOutputs.push_back(p.at("name").get<std::string>());
			else schema.outputs.push_back(jsonToPort(p));
		}
		if (isLogic || !np.signalInputs.empty() || !np.signalOutputs.empty())
			plan.nodes.push_back(std::move(np));

		if (isLogic) continue; // 纯信号面，不入张量数据流

		if (type == "Builtin") {
			// 反序列化时 RunFn 由上层注册，此处仅创建 Schema 骨架
			auto node = std::make_unique<DC::Node>(
				type, name, std::move(schema), nullptr,
				stringToAffinity(j.value("affinity", "Operator")));
			applyNodeMeta(*node, j);
			graph.addNode(std::move(node));
		} else if (reg.hasEngine(type)) {
			// 引擎节点：编译期只物化不加载，见头注释；createLazyNode 以 JSON
			// 声明 schema 调工厂构造；模型加载在宿主绑定期经 loadModel 钩子完成。
			auto node = reg.createLazyNode(type, name, schema);
			if (!node) {
				// 已注册但无工厂：回退骨架，同未注册类型，保留 JSON schema 保证可序列化。
				std::cerr << "GraphCompiler: warning — engine node '" << name
					<< "' (type '" << type << "') has no node factory, "
					<< "creating skeleton (RunFn=nullptr); JSON schema preserved" << std::endl;
				auto skeleton = std::make_unique<DC::Node>(
					type, name, std::move(schema), nullptr,
					stringToAffinity(j.value("affinity", "Operator")));
				applyNodeMeta(*skeleton, j);
				graph.addNode(std::move(skeleton));
				continue;
			}
			// 空声明 schema：节点不可接线，编译期警告。
			if (node->schema().inputs.empty() && node->schema().outputs.empty()) {
				std::cerr << "GraphCompiler: warning — engine node '" << name
					<< "' (type '" << type << "') has empty declared schema: "
					<< "inputs/outputs must be declared in JSON for wiring and execution" << std::endl;
			}
			applyNodeMeta(*node, j);
			graph.addNode(std::move(node));
		} else {
			std::cerr << "GraphCompiler: warning — unregistered engine type '" << type
				<< "' for node '" << name << "', creating skeleton (RunFn=nullptr)" << std::endl;
			auto node = std::make_unique<DC::Node>(
				type, name, std::move(schema), nullptr,
				stringToAffinity(j.value("affinity", "Operator")));
			applyNodeMeta(*node, j);
			graph.addNode(std::move(node));
		}
	}

	// 信号端子键集（边分类用）：node\0port
	std::set<std::string> sigPortKeys;
	for (const auto& n : plan.nodes) {
		for (const auto& p : n.signalInputs) sigPortKeys.insert(n.name + '\0' + p);
		for (const auto& p : n.signalOutputs) sigPortKeys.insert(n.name + '\0' + p);
	}

	// 第二遍：分类边。信号边入 plan；张量边交 rebuildEdges（穿透连接器）。
	nlohmann::json tensorEdges = nlohmann::json::array();
	if (root.contains("edges") && root["edges"].is_array()) {
		for (auto& e : root["edges"]) {
			if (!e.is_object()) continue;
			std::string sn = e.value("srcNode", std::string());
			std::string sp = e.value("srcPort", std::string());
			std::string dn = e.value("dstNode", std::string());
			std::string dp = e.value("dstPort", std::string());
			const bool isSig = sigPortKeys.count(sn + '\0' + sp) || sigPortKeys.count(dn + '\0' + dp);
			if (isSig)
				plan.edges.push_back({sn, sp, dn, dp});
			else
				tensorEdges.push_back(e);
		}
	}
	rebuildEdges(graph, tensorEdges);

	// 第三遍：信号面降级。gate → bindSignal(wire)；const → 编译期写全局 wire。
	// wire 名：源用对外名（宿主直达），其余 node.port。fail-open：gate 无入边则不绑定。
	{
		std::map<std::string, const SignalNodePlan*> byName;
		for (const auto& n : plan.nodes) byName[n.name] = &n;
		auto wireOf = [&](const std::string& node, const std::string& port) -> std::string {
			auto it = byName.find(node);
			if (it != byName.end()) return signalWireName(*it->second, port);
			return node + "." + port;
		};
		auto store = graph.signalStore();
		for (const auto& n : plan.nodes) {
			if (n.isLogic && n.type == kSignalTypeConst && n.hasConstValue)
				store->set(signalWireName(n, kSigOut), n.constValue);
		}
		// 输出信号发布面：计算节点绑定仓库，算法经 RunContext::emitSignal 自决时机发布
		for (const auto& n : plan.nodes) {
			if (n.isLogic) continue;
			auto* node = graph.node(n.name);
			if (node) node->attachSignalStore(store);
		}
		for (const auto& e : plan.edges) {
			if (e.dstPort != kSigGate) continue;
			auto* dst = graph.node(e.dstNode);
			if (!dst) continue; // 逻辑节点无 DC::Node，不被门控
			dst->bindSignal(store, wireOf(e.srcNode, e.srcPort));
		}
	}

	graph.setSignalPlan(std::move(plan));

	// 输出绑定：alias 缺省回退 nodeName.portName，唯一且兼容旧版本文件
	if (root.contains("outputBindings")) {
		for (auto& b : root["outputBindings"]) {
			graph.bindOutput(
				b.value("alias", b.at("nodeName").get<std::string>() + "." + b.at("portName").get<std::string>()),
				b.at("nodeName").get<std::string>(),
				b.at("portName").get<std::string>());
		}
	}

	// 输入绑定：alias 缺省回退 nodeName.portName，唯一且兼容旧版本文件
	if (root.contains("inputBindings")) {
		for (auto& b : root["inputBindings"]) {
			graph.bindInput(
				b.value("alias", b.at("nodeName").get<std::string>() + "." + b.at("portName").get<std::string>()),
				b.at("nodeName").get<std::string>(),
				b.at("portName").get<std::string>());
		}
	}

}

void GraphCompiler::compileFile(InferGraph& graph, std::string_view path) {
	std::filesystem::path p(path);
	std::string ext = p.extension().string();

	if (ext == ".dcg") {
		// .dcg：只读取 graph.json；模型解压与加载由宿主处理，经 DcgArchive::extractOne，
		// 含路径越界、符号链接与预算防御。
		auto archive = DcgArchive::openRead(p);

		// 读取 graph.json，解析前拒绝超限输入
		std::string json = archive->readGraphJson();
		if (json.size() > kMaxGraphJsonBytes) {
			throw GraphException(GraphException::ErrorType::Other,
				"GraphCompiler::compileFile",
				"graph.json in .dcg exceeds size limit (" + std::to_string(json.size()) + " > "
					+ std::to_string(kMaxGraphJsonBytes) + " bytes)");
		}

		nlohmann::json root;
		try {
			root = nlohmann::json::parse(json);
		} catch (const nlohmann::json::exception& e) {
			throw GraphException(GraphException::ErrorType::Other,
				"GraphCompiler::compileFile",
				std::string("JSON parse error in .dcg: ") + e.what());
		}

		// nodes 形状校验：schema 约定为数组；buildGraph 对 object 也能迭代，入口显式拒绝。
		if (root.contains("nodes") && !root["nodes"].is_array()) {
			throw GraphException(GraphException::ErrorType::Other,
				"GraphCompiler::compileFile",
				"invalid .dcg graph.json: 'nodes' must be an array");
		}

		// 构建图；archive 析构自动清理空临时目录
		compileInternal(graph, root);
		return;
	}

	std::ifstream ifs(p, std::ios::binary);
	if (!ifs.is_open()) {
		throw GraphException(GraphException::ErrorType::Other,
							"GraphCompiler::compileFile",
							"cannot open file: " + std::string(path));
	}
	// Read at most 32 MiB + one sentinel byte, including growing/non-seekable
	// inputs. Never trust a pre-read file_size check as the allocation boundary.
	std::string content;
	content.reserve(kMaxGraphJsonBytes + 1);
	char chunk[64 * 1024];
	while (content.size() <= kMaxGraphJsonBytes) {
		const auto request = std::min<std::size_t>(sizeof(chunk), kMaxGraphJsonBytes + 1 - content.size());
		ifs.read(chunk, static_cast<std::streamsize>(request));
		content.append(chunk, static_cast<std::size_t>(ifs.gcount()));
		if (content.size() > kMaxGraphJsonBytes) {
			throw GraphException(GraphException::ErrorType::Other, "GraphCompiler::compileFile",
				"graph definition exceeds size limit (33554432 bytes)");
		}
		if (ifs.bad() || (ifs.fail() && !ifs.eof())) {
			throw GraphException(GraphException::ErrorType::Other, "GraphCompiler::compileFile",
				"failed to read graph definition");
		}
		if (ifs.eof()) break;
	}

	compileString(graph, content);
}

void GraphCompiler::compileString(InferGraph& graph, std::string_view json) {
	// 大小预算：解析前拒绝超限输入，防无界分配
	if (json.size() > kMaxGraphJsonBytes) {
		throw GraphException(GraphException::ErrorType::Other, "GraphCompiler::compileString",
							 "graph definition exceeds size limit (" + std::to_string(json.size()) + " > "
								 + std::to_string(kMaxGraphJsonBytes) + " bytes)");
	}
	try {
		auto root = nlohmann::json::parse(json);
		compileInternal(graph, root);
	} catch (const nlohmann::json::exception& e) {
		throw GraphException(GraphException::ErrorType::Other,
							"GraphCompiler::compileString",
							std::string("JSON parse error: ") + e.what());
	}
}

void GraphCompiler::compileInternal(InferGraph& graph, const nlohmann::json& root) {
	try {
		buildGraph(graph, root);
	} catch (const nlohmann::json::exception& e) {
		throw GraphException(GraphException::ErrorType::Other,
							"GraphCompiler::compileInternal",
							std::string("JSON parse error: ") + e.what());
	}
}

void GraphCompiler::serialize(const InferGraph& graph, std::string_view path) {
	std::filesystem::path p(path);
	std::string ext = p.extension().string();

	if (ext == ".dcg") {
		auto json = graphToJson(graph);

		// 模型文件：原 modelPath 映射到 archive 内路径
		std::map<std::string, std::string> modelFiles;
		std::set<std::string> usedNames;

		for (auto& j : json["nodes"]) {
			if (!j.contains("modelPath")) continue;
			std::string origPath = j["modelPath"].get<std::string>();

			// 同一磁盘文件被多节点引用：复用已分配的 archive 名，只入包一份；
			// 二次改名会覆盖记录导致引用悬空。
			if (auto it = modelFiles.find(origPath); it != modelFiles.end()) {
				j["modelPath"] = it->second;
				continue;
			}

			std::filesystem::path orig(origPath);
			std::string baseName = orig.filename().string();
			std::string archiveName = "models/" + baseName;

			// 不同源路径同 basename：加数字后缀避让
			int suffix = 1;
			while (!usedNames.insert(archiveName).second) {
				archiveName = "models/" + orig.stem().string() + "_" + std::to_string(suffix++)
					+ orig.extension().string();
			}

			modelFiles[origPath] = archiveName;
			j["modelPath"] = archiveName;
		}

		auto archive = DcgArchive::openWrite(p);
		archive->writeGraphJson(json.dump(2));

		for (auto& [origPath, archivePath] : modelFiles) {
			archive->addModelFile(archivePath, origPath);
		}

		archive->finalize();
		return;
	}

	auto json = graphToJson(graph);
	std::string out = json.dump(2);

	std::ofstream ofs(std::string(path), std::ios::binary);
	if (!ofs.is_open()) {
		throw GraphException(GraphException::ErrorType::Other,
							"GraphCompiler::serialize",
							"cannot open file for writing: " + std::string(path));
	}
	ofs << out;
	if (!ofs) {
		throw GraphException(GraphException::ErrorType::Other,
							"GraphCompiler::serialize",
							"failed to write: " + std::string(path));
	}
}

} // namespace DC::Ir
