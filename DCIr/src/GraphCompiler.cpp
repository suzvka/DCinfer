#include "Ir/GraphCompiler.h"

#include "Connector.h"
#include "EngineRegistry.h"
#include "GraphException.h"
#include "Ir/DcgArchive.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>

namespace DC::Ir {

namespace {

// ── 图编译输入预算（P2-13）──
/// JSON 输入总字节上限：不可信/异常来源的图定义在入口拒绝，防无界分配
/// （nlohmann 解析内存与输入体积同量级）。.dcg 路径的解压侧已有逐条目
/// 与聚合预算（DcgArchive），本常量补齐 graph.json 字符串层面的防线。
constexpr std::size_t kMaxGraphJsonBytes = 32ull * 1024 * 1024;
/// 节点数上限：图规模边界（buildGraph 为全部编译路径的必经点）。
constexpr std::size_t kMaxGraphNodes = 4096;

} // namespace

// ════════════════════════════════════════════
// 端口 shape 的类型与 -1 语义
// ════════════════════════════════════════════
// Node::Port::shape 的类型是 Tensor::Shape = std::vector<int64_t>
// （DCinfer/include/Tensor/Tensor.hpp），本身即可表达 ONNX 动态维度 -1，
// 序列化/反序列化直接以 int64_t 直通即可保证 roundtrip 稳定。
// 注意：TensorData::Shape（TensorData.h）= std::vector<size_t> 无法表达
// -1，若以含 -1 的 schema shape 构造实际 TensorData，维度会隐式转换为
// size_t::max 且形状乘积溢出——这是核心库运行期数据路径的已知限制
// （见 GraphCompiler.h 头注释），不在 DCIr 侧解决。

// ════════════════════════════════════════════
// 辅助：affinity 字符串转换
// ════════════════════════════════════════════

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

// ════════════════════════════════════════════
// 辅助：端口 ↔ JSON
// ════════════════════════════════════════════

nlohmann::json GraphCompiler::portToJson(const Node::Port& port) {
	nlohmann::json j;
	j["name"] = port.name;
	j["tensorType"] = TensorMeta::typeToString(port.type);
	j["typeSize"] = static_cast<int64_t>(port.typeSize);
	nlohmann::json shapeArr = nlohmann::json::array();
	for (auto dim : port.shape) {
		// Tensor::Shape = vector<int64_t>：动态维度 -1 原样写入 JSON，roundtrip 稳定
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
	// typeSize 校验（IR-08）：负值经 static_cast<size_t> 会穿透为 SIZE_MAX，
	// 直接被缓冲/张量路径当作巨额单元大小——显式拒绝；0 合法
	// （Void + 0 = 不校验类型语义）。
	const int64_t typeSize = j.at("typeSize").get<int64_t>();
	if (typeSize < 0 || typeSize > (1ll << 20)) {
		throw GraphException(GraphException::ErrorType::Other, "GraphCompiler::jsonToPort",
			"invalid typeSize " + std::to_string(typeSize) + " for port '" + p.name + "'");
	}
	p.typeSize = static_cast<size_t>(typeSize);
	for (auto& dim : j.at("shape")) {
		// 直接以 int64_t 保留（含 -1 动态维度）：禁止 static_cast<size_t> 等
		// 有符号/无符号转换——JSON -1 会被转为巨大值，破坏 roundtrip 对称性。
		p.shape.push_back(dim.get<int64_t>());
	}
	p.required = j.value("required", true);
	return p;
}

// ════════════════════════════════════════════
// 辅助：节点元信息（modelPath / tag）
// ════════════════════════════════════════════

/// @brief 将 JSON 中的 tag / modelPath 应用到已创建节点（Builtin / 引擎 / 骨架三分支共用）
///        modelPath 为引擎自定义不透明信息：原样透传——不拼接、不校验、
///        不做任何文件系统解释（相对路径/URL/模型标识符均原样保留），
///        其消费语义（加载方式与时机）由引擎适配层决定。
static void applyNodeMeta(DC::Node& node, const nlohmann::json& j) {
	if (j.contains("modelPath")) {
		node.setModelPath(j["modelPath"].get<std::string>());
	}
	if (j.contains("tag")) {
		node.setTag(j["tag"].get<std::string>());
	}
}

// ════════════════════════════════════════════
// 序列化辅助：折叠连接器，推断边 mode
// ════════════════════════════════════════════

nlohmann::json GraphCompiler::edgesToJson(const InferGraph& graph) {
	nlohmann::json edgesArr = nlohmann::json::array();

	// 索引：connector 名 → 其所有输出边
	// edges() 返回值副本（P2-13 引用收窄）：connectorOut 存储元素指针，
	// 必须先把快照钉在局部变量上延长生命周期至函数尾，否则 range-for
	// 的临时容器在循环结束后析构，指针全部悬垂
	const auto edgesSnapshot = graph.edges();
	std::map<std::string, std::vector<const InferGraph::Edge*>, std::less<>> connectorOut;
	for (const auto& e : edgesSnapshot) {
		auto* srcNode = graph.node(e.srcNode);
		if (srcNode && srcNode->isConnector()) {
			connectorOut[e.srcNode].push_back(&e);
		}
	}

	// 连接器是否承担多路分发（Broadcast 且出边数 > 1）
	auto isFanOutConnector = [&](const std::string& name) {
		auto* n = graph.node(name);
		if (!n || !n->isConnector() || n->type().find("Broadcast") == std::string::npos) return false;
		auto it = connectorOut.find(name);
		return it != connectorOut.end() && it->second.size() > 1;
	};

	// 遍历 processor 源边：若指向连接器，穿透连接器链（自动导线 / 显式
	// Broadcast / 包裹导线的任意组合），展开为 processor → processor 逻辑边。
	// 链上出现多路分发 Broadcast 时，展开出的每条逻辑边标 mode=broadcast
	// （重建时还原为同一分发组）；纯 1:1 链折叠为无 mode 直连边。
	for (auto& e : graph.edges()) {
		auto* srcNode = graph.node(e.srcNode);
		if (!srcNode || srcNode->isConnector()) continue;
		auto* dstNode = graph.node(e.dstNode);
		if (!dstNode) continue;

		if (!dstNode->isConnector()) {
			// 两处理器直连（保守处理）
			nlohmann::json edge;
			edge["srcNode"] = e.srcNode;
			edge["srcPort"] = e.srcPort;
			edge["dstNode"] = e.dstNode;
			edge["dstPort"] = e.dstPort;
			edgesArr.push_back(std::move(edge));
			continue;
		}

		// 连接器链穿透：迭代收集所有逻辑终点（非连接器节点）
		struct WalkItem {
			std::string nodeName; ///< 当前到达的节点
			std::string portName; ///< 进入该节点的输入口
			bool fanOutPath;      ///< 路径上已出现多路分发 Broadcast
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
				// 逻辑终点：processor → processor
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

			// 中间连接器：继续穿透其所有出边
			auto it = connectorOut.find(item.nodeName);
			if (it == connectorOut.end()) continue; // 悬空连接器：无下游，丢弃
			for (auto* outE : it->second) {
				stack.push_back({outE->dstNode, outE->dstPort,
								 item.fanOutPath || isFanOutConnector(outE->srcNode)});
			}
		}
	}
	return edgesArr;
}

// ════════════════════════════════════════════
// 序列化：InferGraph → JSON
// ════════════════════════════════════════════

nlohmann::json GraphCompiler::graphToJson(const InferGraph& graph) {
	nlohmann::json root;
	root["version"] = "1.0";

	// 节点：只序列化非连接器节点（处理器）
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

		nlohmann::json inputs = nlohmann::json::array();
		for (auto& port : node->schema().inputs) {
			inputs.push_back(portToJson(port));
		}
		j["inputs"] = std::move(inputs);

		nlohmann::json outputs = nlohmann::json::array();
		for (auto& port : node->schema().outputs) {
			outputs.push_back(portToJson(port));
		}
		j["outputs"] = std::move(outputs);

		nodesArr.push_back(std::move(j));
	}
	root["nodes"] = std::move(nodesArr);

	// 边：折叠连接器
	root["edges"] = edgesToJson(graph);

	// 输出绑定（alias 必填：签名元数据字段，不参与运行时寻址）
	nlohmann::json bindingsArr = nlohmann::json::array();
	for (auto& b : graph.outputBindings()) {
		auto* boundNode = graph.node(b.nodeName);
		if (boundNode && boundNode->isConnector()) continue; // 跳过连接器输出绑定
		nlohmann::json jb;
		jb["alias"] = b.alias;
		jb["nodeName"] = b.nodeName;
		jb["portName"] = b.portName;
		bindingsArr.push_back(std::move(jb));
	}
	root["outputBindings"] = std::move(bindingsArr);

	// 输入绑定（alias 必填：签名元数据字段，不参与运行时寻址）
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

// ════════════════════════════════════════════
// 反序列化辅助：按 mode 重建边
// ════════════════════════════════════════════

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
			// Routing 连接器已随核心库移除（轮询属业务语义，应由上层自定义节点实现）：
			// 旧版本序列化的图在此显式报错，而非静默降级为普通连线
			throw DC::GraphException(DC::GraphException::ErrorType::Other,
									 "GraphCompiler::rebuildEdges",
									 "edge mode \"routing\" is no longer supported; express "
									 "round-robin via a custom node with per-instance selection");
		}

		if (key.mode == "broadcast") {
			// 创建 Broadcast(N) 连接器
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

			// src → conn.in；conn.out_i → dst_i（connect 自动包裹直通导线，
			// 序列化折叠后不可见，round-trip 幂等不受影响）
			// 连接失败不再容忍（IR-07）：孤儿连接器 / 残缺图属静默错误，
			// GraphException（含节点/端口坐标）直接透传，反序列化 fail-fast。
			graph.connect(key.srcNode, key.srcPort, connName, "in");
			// conn.out_i → dst_i
			for (size_t i = 0; i < targets.size(); ++i) {
				graph.connect(connName, "out_" + std::to_string(i), targets[i].dstNode, targets[i].dstPort);
			}
		} else {
			// 默认 1→1：用 connect() 自动插入导线连接器
			// 连接失败直接 fail-fast（IR-07，同上）
			for (auto& tgt : targets) {
				graph.connect(key.srcNode, key.srcPort, tgt.dstNode, tgt.dstPort);
			}
		}
	}
}

// ════════════════════════════════════════════
// 反序列化：JSON → InferGraph
// ════════════════════════════════════════════

void GraphCompiler::buildGraph(InferGraph& graph, const nlohmann::json& root) {

	// 节点数预算（P2-13）：全部编译路径（json 字符串/.dcg）的必经点
	if (root.contains("nodes") && root["nodes"].is_array()
		&& root["nodes"].size() > kMaxGraphNodes) {
		throw GraphException(GraphException::ErrorType::Other, "GraphCompiler::buildGraph",
							 "too many nodes in graph definition (" + std::to_string(root["nodes"].size())
								 + " > limit " + std::to_string(kMaxGraphNodes) + ")");
	}

	// 节点
	for (auto& j : root.at("nodes")) {
		std::string name = j.at("name").get<std::string>();
		std::string type = j.at("type").get<std::string>();

		// 解析 Schema
		Node::Schema schema;
		for (auto& p : j.at("inputs")) {
			schema.inputs.push_back(jsonToPort(p));
		}
		for (auto& p : j.at("outputs")) {
			schema.outputs.push_back(jsonToPort(p));
		}

		auto& reg = EngineRegistry::instance();

		if (type == "Builtin") {
			// Builtin 节点：尝试从 Registry 查找已注册算子
			// 反序列化时 RunFn 由上层注册，此处仅创建 Schema 骨架
			auto node = std::make_unique<DC::Node>(
				type, name, std::move(schema), nullptr,
				stringToAffinity(j.value("affinity", "Operator")));
			applyNodeMeta(*node, j);
			graph.addNode(std::move(node));
		} else if (reg.hasEngine(type)) {
			// 引擎节点：编译期只物化、不加载（见 GraphCompiler.h 头注释）。
			// createLazyNode 以 JSON 声明 schema（不做实例推导）调用工厂构造节点；
			// modelPath 原样透传，模型加载由引擎 loadModel 钩子在宿主
			// 绑定期/执行期处理（getOrCreateEngine 先确保引擎核心、再加载模型
			// + Node::bindEngine）。
			auto node = reg.createLazyNode(type, name, schema);
			if (!node) {
				// 引擎已注册但未注册工厂：回退骨架（与未注册类型一致），
				// 保留 JSON 声明 schema，保证图结构完整可序列化。
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
			// 声明 schema 空检查：接线与执行均以节点 schema 为锚，
			// JSON 未声明任何端口时该节点不可接线，编译期给出警告。
			if (node->schema().inputs.empty() && node->schema().outputs.empty()) {
				std::cerr << "GraphCompiler: warning — engine node '" << name
					<< "' (type '" << type << "') has empty declared schema: "
					<< "inputs/outputs must be declared in JSON for wiring and execution" << std::endl;
			}
			applyNodeMeta(*node, j);
			graph.addNode(std::move(node));
		} else {
			// 未注册类型：创建骨架节点（RunFn 留空）
			std::cerr << "GraphCompiler: warning — unregistered engine type '" << type
				<< "' for node '" << name << "', creating skeleton (RunFn=nullptr)" << std::endl;
			auto node = std::make_unique<DC::Node>(
				type, name, std::move(schema), nullptr,
				stringToAffinity(j.value("affinity", "Operator")));
			applyNodeMeta(*node, j);
			graph.addNode(std::move(node));
		}
	}

	// 边
	if (root.contains("edges")) {
		rebuildEdges(graph, root["edges"]);
	}

	// 输出绑定（alias 缺省回退 nodeName.portName：唯一且兼容旧版本序列化文件）
	if (root.contains("outputBindings")) {
		for (auto& b : root["outputBindings"]) {
			graph.bindOutput(
				b.value("alias", b.at("nodeName").get<std::string>() + "." + b.at("portName").get<std::string>()),
				b.at("nodeName").get<std::string>(),
				b.at("portName").get<std::string>());
		}
	}

	// 输入绑定（alias 缺省回退 nodeName.portName：唯一且兼容旧版本序列化文件）
	if (root.contains("inputBindings")) {
		for (auto& b : root["inputBindings"]) {
			graph.bindInput(
				b.value("alias", b.at("nodeName").get<std::string>() + "." + b.at("portName").get<std::string>()),
				b.at("nodeName").get<std::string>(),
				b.at("portName").get<std::string>());
		}
	}

}

// ════════════════════════════════════════════
// 公开接口
// ════════════════════════════════════════════

void GraphCompiler::compileFile(InferGraph& graph, std::string_view path) {
	std::filesystem::path p(path);
	std::string ext = p.extension().string();

	if (ext == ".dcg") {
		// ── .dcg 反序列化：只读取 graph.json ──
		// 模型文件不由编译期解压/加载：modelPath 为归档内相对路径字符串，
		// 原样透传保留在节点上；资源就绪由宿主自行处理（DcgArchive::extractOne
		// 提供含路径越界/符号链接/预算防御的安全解压，详见头注释）。
		auto archive = DcgArchive::openRead(p);

		// 1. 读取并解析 graph.json（大小预算（P2-13）：解析前拒超限输入）
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

		// 2. nodes 形状校验：图描述 schema 约定 nodes 为数组（序列化侧亦如此）；
		//    buildGraph 对 object 形状也能迭代（错误在深处才暴露），入口显式拒绝。
		if (root.contains("nodes") && !root["nodes"].is_array()) {
			throw GraphException(GraphException::ErrorType::Other,
				"GraphCompiler::compileFile",
				"invalid .dcg graph.json: 'nodes' must be an array");
		}

		// 3. 构建图（archive 析构时自动清理空临时目录）
		compileInternal(graph, root);
		return;
	}

	// ── .json 反序列化 ──

	// 读取文件内容
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
	// 大小预算（P2-13）：不可信来源的图定义在解析前拒绝，防无界分配
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
		// ── .dcg 序列化 ──
		auto json = graphToJson(graph);

		// 收集所有模型文件：原 modelPath → archive 内路径
		std::map<std::string, std::string> modelFiles; // original path → archive path
		std::set<std::string> usedNames;

		for (auto& j : json["nodes"]) {
			if (!j.contains("modelPath")) continue;
			std::string origPath = j["modelPath"].get<std::string>();

			// 共享模型（IR-01）：同一磁盘文件被多个节点引用时复用已分配的
			// archive 名——只入包一份、所有引用节点写回同一相对路径。
			// （重复条目二次改名会覆盖 modelFiles 记录 → 首节点 graph.json
			// 引用悬空、.dcg 编译必然失败）
			if (auto it = modelFiles.find(origPath); it != modelFiles.end()) {
				j["modelPath"] = it->second;
				continue;
			}

			// 生成 archive 内唯一名称: models/<basename>
			std::filesystem::path orig(origPath);
			std::string baseName = orig.filename().string();
			std::string archiveName = "models/" + baseName;

			// 同名冲突（不同源路径同 basename）：加数字后缀
			int suffix = 1;
			while (!usedNames.insert(archiveName).second) {
				archiveName = "models/" + orig.stem().string() + "_" + std::to_string(suffix++)
					+ orig.extension().string();
			}

			modelFiles[origPath] = archiveName;
			// 将 modelPath 替换为相对路径
			j["modelPath"] = archiveName;
		}

		// 写入 ZIP
		auto archive = DcgArchive::openWrite(p);
		archive->writeGraphJson(json.dump(2));

		for (auto& [origPath, archivePath] : modelFiles) {
			archive->addModelFile(archivePath, origPath);
		}

		archive->finalize();
		return;
	}

	// ── .json 序列化 ──
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
