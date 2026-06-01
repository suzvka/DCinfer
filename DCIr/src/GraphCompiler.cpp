#include "Ir/GraphCompiler.h"

#include "Connector.h"
#include "EngineRegistry.h"
#include "GraphException.h"
#include "Ir/DcgArchive.h"

#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>

namespace DC::Ir {

// ════════════════════════════════════════════
// 辅助：affinity 字符串转换
// ════════════════════════════════════════════

static std::string affinityToString(ThreadPoolAffinity a) {
	switch (a) {
	case ThreadPoolAffinity::Compute: return "Compute";
	case ThreadPoolAffinity::Operator: return "Operator";
	case ThreadPoolAffinity::System: return "System";
	}
	return "Operator";
}

static ThreadPoolAffinity stringToAffinity(const std::string& s) {
	if (s == "Compute") return ThreadPoolAffinity::Compute;
	if (s == "Operator") return ThreadPoolAffinity::Operator;
	if (s == "System") return ThreadPoolAffinity::System;
	return ThreadPoolAffinity::Operator;
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
		shapeArr.push_back(static_cast<int64_t>(dim));
	}
	j["shape"] = std::move(shapeArr);
	j["required"] = port.required;
	return j;
}

static Node::Port jsonToPort(const nlohmann::json& j) {
	Node::Port p;
	p.name = j.at("name").get<std::string>();
	p.type = TensorMeta::stringToType(j.at("tensorType").get<std::string>());
	p.typeSize = static_cast<size_t>(j.at("typeSize").get<int64_t>());
	for (auto& dim : j.at("shape")) {
		p.shape.push_back(static_cast<size_t>(dim.get<int64_t>()));
	}
	p.required = j.value("required", true);
	return p;
}

// ════════════════════════════════════════════
// 序列化辅助：折叠连接器，推断边 mode
// ════════════════════════════════════════════

nlohmann::json GraphCompiler::edgesToJson(const InferGraph& graph) {
	nlohmann::json edgesArr = nlohmann::json::array();

	// 索引：connector 名 → 其所有输出边
	std::map<std::string, std::vector<const InferGraph::Edge*>, std::less<>> connectorOut;
	for (auto& e : graph.edges()) {
		auto* srcNode = graph.node(e.srcNode);
		if (srcNode && srcNode->isConnector()) {
			connectorOut[e.srcNode].push_back(&e);
		}
	}

	// 遍历 processor → connector 边，展开为 processor → processor
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

		// dstNode 是连接器：查下游
		const std::string& cType = dstNode->type();
		std::string mode;
		if (cType.find("Routing") != std::string::npos) {
			mode = "routing";
		} else if (cType.find("Broadcast") != std::string::npos) {
			auto it = connectorOut.find(e.dstNode);
			if (it != connectorOut.end() && it->second.size() > 1) {
				mode = "broadcast";
			}
			// N=1 → 不输出 mode（反序列化用 wire 还原）
		}

		auto it = connectorOut.find(e.dstNode);
		if (it == connectorOut.end()) continue;
		for (auto* outE : it->second) {
			nlohmann::json edge;
			edge["srcNode"] = e.srcNode;
			edge["srcPort"] = e.srcPort;
			edge["dstNode"] = outE->dstNode;
			edge["dstPort"] = outE->dstPort;
			if (!mode.empty()) {
				edge["mode"] = mode;
			}
			edgesArr.push_back(std::move(edge));
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

	// 输出绑定
	nlohmann::json bindingsArr = nlohmann::json::array();
	for (auto& b : graph.outputBindings()) {
		auto* boundNode = graph.node(b.nodeName);
		if (boundNode && boundNode->isConnector()) continue; // 跳过连接器输出绑定
		nlohmann::json jb;
		jb["nodeName"] = b.nodeName;
		jb["portName"] = b.portName;
		bindingsArr.push_back(std::move(jb));
	}
	root["outputBindings"] = std::move(bindingsArr);

	// 输入绑定
	nlohmann::json inputBindingsArr = nlohmann::json::array();
	for (auto& b : graph.inputBindings()) {
		auto* boundNode = graph.node(b.nodeName);
		if (boundNode && boundNode->isConnector()) continue;
		nlohmann::json jb;
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

		if (key.mode == "broadcast" || key.mode == "routing") {
			// 创建 Broadcast(N) 或 Routing(N) 连接器
			size_t n = targets.size();
			Node::Schema connSchema;
			Node::RunFn connRunFn;
			std::string connType;
			if (key.mode == "broadcast") {
				connSchema = DC::Connector::broadcastSchema(n);
				connRunFn = DC::Connector::broadcastRunFn();
				connType = "Connector.Broadcast";
			} else {
				connSchema = DC::Connector::routingSchema(n);
				connRunFn = DC::Connector::routingRunFn();
				connType = "Connector.Routing";
			}
			std::string connName = "__" + key.mode + "_" + std::to_string(connId++);

			auto connNode = std::make_unique<DC::Node>(
				connType, connName, std::move(connSchema), std::move(connRunFn),
				nullptr, ThreadPoolAffinity::System);
			connNode->setConnector(true);
			graph.addNode(std::move(connNode));

			// src → conn.in
			if (!graph.connect(key.srcNode, key.srcPort, connName, "in")) {
				std::cerr << "GraphCompiler: warning — failed to connect '" << key.srcNode
					<< "." << key.srcPort << "' → '" << connName << ".in'" << std::endl;
			}
			// conn.out_i → dst_i
			for (size_t i = 0; i < targets.size(); ++i) {
				std::string outPort = "out_" + std::to_string(i);
				if (!graph.connect(connName, outPort, targets[i].dstNode, targets[i].dstPort)) {
					std::cerr << "GraphCompiler: warning — failed to connect '" << connName
						<< "." << outPort << "' → '" << targets[i].dstNode
						<< "." << targets[i].dstPort << "'" << std::endl;
				}
			}
		} else {
			// 默认 1→1：用 wire() 自动插入导线连接器
			for (auto& tgt : targets) {
				if (!graph.wire(key.srcNode, key.srcPort, tgt.dstNode, tgt.dstPort)) {
					std::cerr << "GraphCompiler: warning — failed to wire '" << key.srcNode
						<< "." << key.srcPort << "' → '" << tgt.dstNode
						<< "." << tgt.dstPort << "'" << std::endl;
				}
			}
		}
	}
}

// ════════════════════════════════════════════
// 反序列化：JSON → InferGraph
// ════════════════════════════════════════════

void GraphCompiler::buildGraph(InferGraph& graph, const nlohmann::json& root, const std::filesystem::path& baseDir) {

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
				type, name, std::move(schema), nullptr, nullptr,
				stringToAffinity(j.value("affinity", "Operator")));
			if (j.contains("tag")) {
				node->setTag(j["tag"].get<std::string>());
			}
			if (j.contains("modelPath")) {
				std::string mp = j["modelPath"].get<std::string>();
				std::filesystem::path mpPath(mp);
				if (mpPath.is_relative()) {
					mpPath = baseDir / mpPath;
					mp = mpPath.string();
				}
				node->setModelPath(std::move(mp));
			}
			graph.addNode(std::move(node));
		} else if (reg.hasEngine(type)) {
			// 引擎节点
			std::string modelPath;
			if (j.contains("modelPath")) {
				modelPath = j["modelPath"].get<std::string>();
				// 相对路径拼接 baseDir
				std::filesystem::path mp(modelPath);
				if (mp.is_relative()) {
					mp = baseDir / mp;
					modelPath = mp.string();
				}
			}
			auto node = reg.createNode(type, name, modelPath);
			if (node) {
				if (j.contains("tag")) {
					node->setTag(j["tag"].get<std::string>());
				}
				graph.addNode(std::move(node));
			}
		} else {
			// 未注册类型：创建骨架节点（RunFn 留空）
			std::cerr << "GraphCompiler: warning — unregistered engine type '" << type
				<< "' for node '" << name << "', creating skeleton (RunFn=nullptr)" << std::endl;
			auto node = std::make_unique<DC::Node>(
				type, name, std::move(schema), nullptr, nullptr,
				stringToAffinity(j.value("affinity", "Operator")));
			if (j.contains("modelPath")) {
				std::string mp = j["modelPath"].get<std::string>();
				std::filesystem::path mpPath(mp);
				if (mpPath.is_relative()) {
					mpPath = baseDir / mpPath;
					mp = mpPath.string();
				}
				node->setModelPath(std::move(mp));
			}
			if (j.contains("tag")) {
				node->setTag(j["tag"].get<std::string>());
			}
			graph.addNode(std::move(node));
		}
	}

	// 边
	if (root.contains("edges")) {
		rebuildEdges(graph, root["edges"]);
	}

	// 输出绑定
	if (root.contains("outputBindings")) {
		for (auto& b : root["outputBindings"]) {
			graph.bindOutput(
				b.at("nodeName").get<std::string>(),
				b.at("portName").get<std::string>());
		}
	}

	// 输入绑定
	if (root.contains("inputBindings")) {
		for (auto& b : root["inputBindings"]) {
			graph.bindInput(
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
		// ── .dcg 反序列化 ──
		auto archive = DcgArchive::openRead(p);

		// 1. 读取并解析 graph.json
		std::string json = archive->readGraphJson();

		// 2. 解析 JSON 找出所有 modelPath，批量解压到临时目录
		//    这样 buildGraph → createNode → getOrCreateEngine 能找到模型文件
		try {
			auto root = nlohmann::json::parse(json);
			if (root.contains("nodes") && root["nodes"].is_array()) {
				std::set<std::string> extracted;
				for (auto& j : root["nodes"]) {
					if (j.contains("modelPath")) {
						std::string mp = j["modelPath"].get<std::string>();
						if (extracted.insert(mp).second) {
							archive->extractOne(mp);
						}
					}
				}
			}
		} catch (const nlohmann::json::exception& e) {
			throw GraphException(GraphException::ErrorType::Other,
				"GraphCompiler::compileFile",
				std::string("JSON parse error in .dcg: ") + e.what());
		}

		// 3. 构建图（baseDir = 临时目录，相对路径 models/xxx 自动解析）
		compileString(graph, json, archive->tempDir());

		// 4. 引擎已加载模型，清理临时文件
		//    （逐个删除 models/ 下的文件，最后删除临时目录）
		std::error_code ec;
		for (auto& entry : std::filesystem::recursive_directory_iterator(archive->tempDir(), ec)) {
			if (entry.is_regular_file()) {
				archive->cleanup(entry.path());
			}
		}
		return;
	}

	// ── .json 反序列化（原有逻辑）──

	// 读取文件内容
	std::ifstream ifs(p, std::ios::binary);
	if (!ifs.is_open()) {
		throw GraphException(GraphException::ErrorType::Other,
							"GraphCompiler::compileFile",
							"cannot open file: " + std::string(path));
	}
	std::ostringstream oss;
	oss << ifs.rdbuf();
	std::string content = oss.str();

	// baseDir = 文件所在目录
	std::filesystem::path baseDir = p.parent_path();

	compileString(graph, content, baseDir);
}

void GraphCompiler::compileString(InferGraph& graph, std::string_view json, std::filesystem::path baseDir) {
	try {
		auto root = nlohmann::json::parse(json);
		buildGraph(graph, root, baseDir);
	} catch (const nlohmann::json::exception& e) {
		throw GraphException(GraphException::ErrorType::Other,
							"GraphCompiler::compileString",
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

			// 生成 archive 内唯一名称: models/<basename>
			std::filesystem::path orig(origPath);
			std::string baseName = orig.filename().string();
			std::string archiveName = "models/" + baseName;

			// 同名冲突：加数字后缀
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

	// ── .json 序列化（原有逻辑）──
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
